import os
import warnings
from pathlib import Path
from typing import Annotated, TypedDict, Sequence, Optional, List, Literal
from dotenv import load_dotenv

# Qdrant & LangChain 관련 임포트
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_community.tools import TavilySearchResults
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Any

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 환경 변수 로드: 실행 위치(CWD)와 무관하게 이 파일과 같은 폴더의 .env를 사용
_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH)


# ===========================
# State 정의
# ===========================
class AgentState(TypedDict):
    """LangGraph Agent의 상태를 정의하는 TypedDict"""
    # 대화 히스토리
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 사용자 질문
    user_query: str
    # 질문 분석 결과
    # {category, needs_clarification, needs_case_law, clarification_question}
    query_analysis: Optional[dict]
    # 검색 결과 (Document 리스트)
    retrieved_docs: Optional[List[Document]]
    # 웹 검색으로 찾은 판례 정보
    case_law_results: Optional[List[dict]]
    # 생성된 답변
    generated_answer: Optional[str]
    # 현재 라우팅 결정
    next_action: Optional[str]


# ===========================
# Reranker 정의
# ===========================
class JinaReranker(BaseDocumentCompressor):
    model_name: str = "jinaai/jina-reranker-v2-base-multilingual"
    top_n: int = 5
    model: Any = None
    tokenizer: Any = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, trust_remote_code=True, dtype="auto"
        )
        self.model.eval()

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Optional[Any] = None
    ) -> Sequence[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            # Sigmoid 적용하여 0~1 사이 확률로 변환
            scores = self.model(**inputs).logits.squeeze(-1).float().cpu()
            scores = torch.sigmoid(scores).tolist()
            if not isinstance(scores, list):
                scores = [scores]

        # Sort and select top_n
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :self.top_n]

        final_docs = []
        for i in top_indices:
            doc = documents[i]
            doc.metadata["relevance_score"] = scores[i]
            final_docs.append(doc)

        return final_docs


# ===========================
# 노드 함수 정의 (LangGraph 영역)
# ===========================

# Pydantic 모델: 질문 분석 결과
class QueryAnalysis(BaseModel):
    """LLM이 반환할 질문 분석 결과"""
    category: str = Field(description="법률 분야: 노동법, 형사법, 민사법, 기타 중 하나")
    needs_clarification: bool = Field(
        default=False, description="질문이 극도로 모호하여 답변 불가능한지")
    needs_case_law: bool = Field(default=False, description="대법원 판례 검색이 필요한지")
    clarification_question: str = Field(
        default="", description="명확화 필요 시 사용자에게 물어볼 질문")


def create_analyze_query_node(llm: ChatOpenAI):
    """노드 1: 질문 분석 (Structured Output 사용)"""

    # Structured Output을 위한 LLM
    structured_llm = llm.with_structured_output(QueryAnalysis)

    analyze_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 질문을 분석하는 전문가입니다.

1. category: 질문의 법률 분야
   - "노동법": 근로기준법, 임금, 퇴직금, 해고, 산재, 주휴수당 등
   - "형사법": 범죄, 형벌, 수사, 재판, 고소/고발 등
   - "민사법": 계약, 손해배상, 소유권, 채권 등
   - "기타": 위 카테고리에 속하지 않는 법률 질문

2. needs_clarification: 질문이 극도로 모호하여 어떤 답변도 불가능한지 (true/false)
   - true: "법률 질문이요", "도와주세요", "계약" 처럼 1~2단어만 있는 경우
   - false (대부분): 상황이 조금이라도 설명되어 있으면 답변 가능
   - 예: "주15시간 이상 근무했는데 주휴수당을 안 줘" → false (답변 가능)
   - 예: "해고당했어요" → false (부당해고 일반론 설명 가능)

3. needs_case_law: 대법원 판례가 필요한지 (true/false)
   - true: "판례", "판결", "대법원" 등을 명시적으로 언급하거나, 법적 해석이 필요한 쟁점 사안
   - false: 단순 법령 조회, 절차/서식 문의

4. clarification_question: needs_clarification이 true일 때만 작성"""),
        ("human", "{query}")
    ])

    def analyze_query(state: AgentState) -> AgentState:
        """질문 분석 노드: Structured Output으로 분류/명확화/판례 필요 여부 판단"""
        query = state["user_query"]

        print(f"🔎 [질문 분석 중...]")

        chain = analyze_prompt | structured_llm
        analysis: QueryAnalysis = chain.invoke({"query": query})

        print(f"📋 [분석 결과] 분야: {analysis.category}")
        print(f"   명확화 필요: {'예' if analysis.needs_clarification else '아니오'}")
        print(f"   판례 필요: {'예' if analysis.needs_case_law else '아니오'}")

        return {
            "query_analysis": analysis.model_dump()
        }

    return analyze_query


def create_clarify_node(llm: ChatOpenAI):
    """노드 2: 사용자에게 명확화 요청"""

    def request_clarification(state: AgentState) -> AgentState:
        """명확화 요청 노드: 모호한 질문에 대해 구체적인 정보 요청"""
        analysis = state.get("query_analysis", {})
        clarification_q = analysis.get("clarification_question", "")

        if not clarification_q:
            # 기본 명확화 질문
            clarification_q = "질문을 좀 더 구체적으로 해주시겠어요? 어떤 상황인지, 무엇이 궁금하신지 자세히 알려주시면 더 정확한 답변을 드릴 수 있습니다."

        print(f"❓ [명확화 요청]")

        # 친절한 형식으로 답변 구성
        answer = f"""안녕하세요! 질문을 잘 이해하기 위해 몇 가지 확인이 필요합니다.

{clarification_q}

위 내용을 포함해서 다시 질문해 주시면, 더 정확하고 도움이 되는 답변을 드릴 수 있습니다. 😊"""

        return {
            "generated_answer": answer,
            "next_action": "end"
        }

    return request_clarification


def create_search_node(vectorstore: QdrantVectorStore):
    """노드 3: Qdrant 벡터DB 검색"""

    def search_documents(state: AgentState) -> AgentState:
        """검색 실행 노드: Qdrant에서 관련 법령/문서 검색"""
        query = state["user_query"]
        analysis = state.get("query_analysis", {})
        category = analysis.get("category", "기타")

        print(f"🔍 [법령 검색] 쿼리: {query[:50]}...")

        # 카테고리에 따른 검색 최적화 (향후 필터 추가 가능)
        # 1. 1차 검색 (유사도 기반, 더 넓게 검색)
        results = vectorstore.similarity_search_with_score(query, k=20)

        if results:
            docs = [doc for doc, score in results]

            # 2. 리랭킹 (Jina Reranker)
            print(f"🔄 [리랭킹] Jina Reranker로 상위 5개 문서 선별 중...")
            try:
                reranker = JinaReranker(top_n=5)
                reranked_docs = reranker.compress_documents(docs, query)

                if reranked_docs:
                    print(f"✅ [리랭킹 완료] {len(reranked_docs)}개 문서 선별")
                    # 리랭킹 점수 출력
                    for i, doc in enumerate(reranked_docs, 1):
                        print(
                            f"   [{i}] 점수: {doc.metadata.get('relevance_score', 0):.4f} | {doc.page_content[:30]}...")
                    docs = reranked_docs
                else:
                    print(f"⚠️  [리랭킹 결과 없음] 원본 검색 결과 사용 (상위 5개)")
                    docs = docs[:5]
            except Exception as e:
                print(f"⚠️  [리랭킹 오류] {e}")
                print(f"   원본 검색 결과 사용 (상위 5개)")
                docs = docs[:5]

            # avg_score logic updated for re-ranking scores
            if docs:
                scores = [doc.metadata.get("relevance_score", 0)
                          for doc in docs]
                avg_score = sum(scores) / len(scores) if scores else 0.0
            else:
                avg_score = 0.0

            print(f"✅ [검색 최종 완료] {len(docs)}개 문서")
        else:
            docs = []
            print(f"⚠️  [검색 결과 없음]")

        return {
            "retrieved_docs": docs
        }

    return search_documents


def create_case_law_search_node(llm: ChatOpenAI):
    """노드 4: 웹 검색을 통한 대법원 판례 검색"""

    def search_case_law(state: AgentState) -> AgentState:
        """대법원 판례 검색 노드: Tavily를 통해 관련 판례 웹 검색"""
        query = state["user_query"]
        analysis = state.get("query_analysis", {})
        category = analysis.get("category", "기타")

        print(f"⚖️  [판례 검색] 대법원 판례 웹 검색 중...")

        # Tavily API 키 확인
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print(f"⚠️  [판례 검색 스킵] TAVILY_API_KEY가 설정되지 않았습니다.")
            return {"case_law_results": []}

        try:
            # Tavily 검색 도구 설정
            search_tool = TavilySearchResults(
                max_results=3,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False
            )

            # 판례 검색 쿼리 최적화
            search_query = f"대법원 판례 {category} {query}"

            # 검색 실행
            results = search_tool.invoke({"query": search_query})

            if results:
                case_laws = []
                for r in results:
                    case_laws.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:500]  # 내용 제한
                    })
                print(f"✅ [판례 검색 완료] {len(case_laws)}건 발견")
                return {"case_law_results": case_laws}
            else:
                print(f"⚠️  [판례 검색] 관련 판례를 찾지 못했습니다.")
                return {"case_law_results": []}

        except Exception as e:
            print(f"⚠️  [판례 검색 오류] {e}")
            return {"case_law_results": []}

    return search_case_law


def create_generate_node(llm: ChatOpenAI):
    """노드 5: 최종 답변 생성"""

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 전문 AI 어시스턴트 'A-TEAM 봇'입니다.

역할:
- 검색된 법률 문서와 판례를 바탕으로 정확하고 친절하게 답변합니다.
- 법령명, 조항, 판례번호 등 구체적인 근거를 제시합니다.
- 법률 용어는 쉽게 풀어서 설명합니다.

답변 작성 규칙:
1. 검색된 자료를 근거로 답변하세요.
2. 답변 구조: 📌 결론 → 📖 법적 근거 → 💡 추가 설명
3. 관련 법령과 조항을 [법령명 제X조]처럼 명시하세요.
4. 판례가 있으면 [대법원 XXXX. X. X. 선고 XXX다XXXX 판결] 형식으로 인용하세요.
5. 확실하지 않은 내용은 "~로 해석될 수 있습니다" 등으로 신중하게 표현하세요.
6. 전문 법률 상담이 필요한 경우 안내하세요.
7. 한국어로 답변하세요."""),
        ("human", """질문 분야: {category}

사용자 질문: {query}

📚 검색된 법령/문서:
{context}

⚖️ 관련 판례 (웹 검색):
{case_law}

위 자료를 바탕으로 질문에 답변해주세요.""")
    ])

    def generate_answer(state: AgentState) -> AgentState:
        """답변 생성 노드: 검색 결과와 판례를 종합하여 답변 생성"""
        query = state["user_query"]
        analysis = state.get("query_analysis", {})
        category = analysis.get("category", "기타")
        docs = state.get("retrieved_docs", [])
        case_laws = state.get("case_law_results", [])

        print(f"💬 [답변 생성 중...]")

        # 문서 컨텍스트 포맷팅
        if docs:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                source = metadata.get("source", "")
                law_name = metadata.get("law_name", "")
                article = metadata.get("article_no", "")
                title = metadata.get(
                    "article_title", "") or metadata.get("title", "")
                content = doc.page_content[:800]

                header = f"[문서 {i}]"
                if law_name:
                    header += f" {law_name}"
                    if article:
                        header += f" 제{article}조"
                if title:
                    header += f" - {title}"

                context_parts.append(f"{header}\n{content}\n")

            context = "\n".join(context_parts)
        else:
            context = "(관련 법령 문서가 검색되지 않았습니다)"

        # 판례 컨텍스트 포맷팅
        if case_laws:
            case_parts = []
            for i, case in enumerate(case_laws, 1):
                case_parts.append(
                    f"[판례 {i}] {case.get('title', '')}\n{case.get('content', '')}\n출처: {case.get('url', '')}\n")
            case_law_context = "\n".join(case_parts)
        else:
            case_law_context = "(관련 판례 정보 없음)"

        # 검색 결과가 전혀 없는 경우
        if not docs and not case_laws:
            answer = """죄송합니다. 질문과 관련된 법률 정보를 찾지 못했습니다.

다음과 같이 시도해 보시겠어요?
1. 질문을 더 구체적으로 작성해 주세요 (예: 상황, 관련 법령 등)
2. 다른 키워드로 질문해 보세요
3. 복잡한 사안의 경우 전문 법률 상담을 권장드립니다."""
        else:
            # LLM으로 답변 생성
            chain = answer_prompt | llm
            response = chain.invoke({
                "category": category,
                "query": query,
                "context": context,
                "case_law": case_law_context
            })
            answer = response.content

        print(f"✅ [답변 생성 완료]")

        return {
            "generated_answer": answer
        }

    return generate_answer


# ===========================
# 라우팅 함수 (조건부 분기)
# ===========================

def route_after_analysis(state: AgentState) -> Literal["clarify", "search"]:
    """분석 후 라우팅: 명확화 필요 여부에 따라 분기"""
    analysis = state.get("query_analysis", {})
    needs_clarification = analysis.get("needs_clarification", False)

    if needs_clarification:
        return "clarify"
    else:
        return "search"


def route_after_search(state: AgentState) -> Literal["case_law_search", "generate"]:
    """검색 후 라우팅: 판례 필요 여부에 따라 분기"""
    analysis = state.get("query_analysis", {})
    needs_case_law = analysis.get("needs_case_law", False)

    if needs_case_law:
        return "case_law_search"
    else:
        return "generate"


# ===========================
# 사전 준비 영역: 리소스 초기화
# ===========================
def initialize_resources():
    """임베딩 모델, 벡터스토어 초기화"""

    # 1. 환경 변수 로드
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY가 .env 파일에 설정되지 않았습니다!")

    print(f"🔧 설정 로드 완료")

    # 2. 임베딩 모델 설정
    print(f"\n🚀 임베딩 모델 로드 중 (Qwen/Qwen3-Embedding-0.6B)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로드 완료")

    # 3. Qdrant 클라이언트 연결
    print(f"\n📡 Qdrant 연결 중...")
    warnings.filterwarnings(
        'ignore', message='Api key is used with an insecure connection')

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
        prefer_grpc=False)
    print("✅ Qdrant 연결 완료")

    # 4. 벡터스토어 생성
    print(f"\n🗂️  벡터스토어 초기화 중...")
    print("   (컬렉션 검증 중... 네트워크 상태에 따라 시간이 걸릴 수 있습니다)")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="text"
    )
    print("✅ 벡터스토어 초기화 완료")

    return {
        "embeddings": embeddings,
        "vectorstore": vectorstore
    }


# ===========================
# LangGraph 초기화
# ===========================
def initialize_langgraph_chatbot():
    """LangGraph 기반 RAG 챗봇 초기화 (조건부 분기 포함)"""

    # 사전 준비: 리소스 초기화
    resources = initialize_resources()
    vectorstore = resources["vectorstore"]

    # LLM 설정
    print(f"\n🤖 LLM 설정 중...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )
    print("✅ LLM 설정 완료")

    # 노드 생성
    print(f"\n⚙️  LangGraph 노드 생성 중...")
    analyze_node = create_analyze_query_node(llm)
    clarify_node = create_clarify_node(llm)
    search_node = create_search_node(vectorstore)
    case_law_node = create_case_law_search_node(llm)
    generate_node = create_generate_node(llm)
    print("✅ 노드 생성 완료 (5개)")

    # StateGraph 구성
    print(f"\n🔗 LangGraph 워크플로우 구성 중...")
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("search", search_node)
    workflow.add_node("case_law_search", case_law_node)
    workflow.add_node("generate", generate_node)

    # 엣지 추가
    workflow.set_entry_point("analyze")

    # 조건부 분기 1: 분석 후 → 명확화 필요? → clarify / search
    workflow.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "clarify": "clarify",
            "search": "search"
        }
    )

    # clarify는 바로 종료
    workflow.add_edge("clarify", END)

    # 조건부 분기 2: 검색 후 → 판례 필요? → case_law_search / generate
    workflow.add_conditional_edges(
        "search",
        route_after_search,
        {
            "case_law_search": "case_law_search",
            "generate": "generate"
        }
    )

    # 판례 검색 후 → 답변 생성
    workflow.add_edge("case_law_search", "generate")

    # 답변 생성 후 → 종료
    workflow.add_edge("generate", END)

    # 그래프 컴파일
    graph = workflow.compile()
    print("✅ LangGraph 구성 완료")

    return graph


# ===========================
# 메인 실행 함수
# ===========================
def main():
    """LangGraph RAG 챗봇 실행 메인 함수"""

    # API Key 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("💡 .env 파일에 OPENAI_API_KEY를 추가하세요.")
        return

    # Tavily API Key 확인 (경고만)
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  경고: TAVILY_API_KEY가 설정되지 않았습니다.")
        print("   판례 웹 검색 기능이 비활성화됩니다.\n")

    try:
        # 챗봇 초기화
        print("\n" + "="*60)
        print("🚀 A-TEAM 법률 RAG 챗봇 (LangGraph V1) 초기화 시작")
        print("="*60 + "\n")

        graph = initialize_langgraph_chatbot()

        print("\n" + "="*60)
        print("✅ 🤖 A-TEAM 법률 챗봇 준비 완료!")
        print("="*60)
        print("\n💡 사용 방법:")
        print("  - 노동법, 형사법, 민사법 관련 질문에 응답합니다.")
        print("  - 판례가 필요하면 자동으로 웹 검색합니다.")
        print("  - 질문이 모호하면 구체화를 요청합니다.")
        print("  - 'exit', 'quit', '종료'를 입력하면 종료됩니다")
        print("\n📊 워크플로우:")
        print("  ┌─ 질문 분석 ─┬─ [모호함] → 명확화 요청 → 종료")
        print("  │            └─ [명확함] → 법령 검색 ─┬─ [판례 필요] → 판례 검색 → 답변 생성")
        print("  │                                     └─ [불필요] → 답변 생성")
        print("="*60 + "\n")

        # 대화 루프
        while True:
            try:
                # 사용자 입력
                user_input = input("👤 User >> ").strip()

                # 종료 명령 확인
                if user_input.lower() in ["exit", "quit", "종료", "q"]:
                    print("\n👋 챗봇을 종료합니다. 감사합니다!")
                    break

                # 빈 입력 체크
                if not user_input:
                    print("❌ 질문을 입력해주세요.\n")
                    continue

                # 초기 상태 설정
                initial_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_query": user_input,
                    "query_analysis": None,
                    "retrieved_docs": None,
                    "case_law_results": None,
                    "generated_answer": None,
                    "next_action": None
                }

                # 그래프 실행
                print("\n" + "-"*60)
                print("🔄 워크플로우 실행 중...")
                print("-"*60 + "\n")

                result = graph.invoke(initial_state)

                # 최종 답변 출력
                answer = result.get("generated_answer", "")
                if answer:
                    print("\n" + "="*60)
                    print("🤖 AI 답변:")
                    print("="*60)
                    print(f"\n{answer}\n")
                    print("="*60 + "\n")
                else:
                    print("\n⚠️ 답변을 생성할 수 없습니다.\n")

            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다. 감사합니다!")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("💡 다시 시도해주세요.\n")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"\n❌ 챗봇 초기화 실패: {e}")
        print("💡 설정을 확인하고 다시 시도해주세요.")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
