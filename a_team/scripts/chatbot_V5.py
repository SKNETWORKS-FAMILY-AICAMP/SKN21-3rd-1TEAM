################################################
# A-TEAM 법률 RAG 챗봇 (LangGraph V5)
# - 검색 다중 쿼리 + rerank + 노동법 비법령 문서 가중치
# - 근거 스니펫을 정돈된 bullet로 전달해 인용 강제
# - Top-K 소폭 상향, 컨텍스트 길이 제한
################################################

import os
import re
import warnings
from pathlib import Path
from typing import Annotated, TypedDict, Sequence, Optional, List, Literal, Any
from dotenv import load_dotenv

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

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    query_analysis: Optional[dict]
    retrieved_docs: Optional[List[Document]]
    case_law_results: Optional[List[dict]]
    generated_answer: Optional[str]
    next_action: Optional[str]


class JinaReranker(BaseDocumentCompressor):
    model_name: str = "jinaai/jina-reranker-v2-base-multilingual"
    top_n: int = 6
    model: Any = None
    tokenizer: Any = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, trust_remote_code=True, dtype="auto"
        )
        self.model.eval()

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Optional[Any] = None) -> Sequence[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = self.model(**inputs).logits.squeeze(-1).float().cpu()
            scores = torch.sigmoid(scores).tolist()
            if not isinstance(scores, list):
                scores = [scores]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_n]
        final_docs = []
        for i in top_indices:
            doc = documents[i]
            doc.metadata["relevance_score"] = scores[i]
            final_docs.append(doc)
        return final_docs


class QueryAnalysis(BaseModel):
    category: str = Field(description="법률 분야: 노동법, 형사법, 민사법, 기타 중 하나")
    needs_clarification: bool = Field(default=False, description="질문이 극도로 모호하여 답변 불가능한지")
    needs_case_law: bool = Field(default=False, description="대법원 판례 검색이 필요한지")
    clarification_question: str = Field(default="", description="명확화 필요 시 사용자에게 물어볼 질문")


def create_analyze_query_node(llm: ChatOpenAI):
    structured_llm = llm.with_structured_output(QueryAnalysis)
    analyze_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 법률 질문을 분석하는 전문가입니다.

1. category: 질문의 법률 분야
   - "노동법": 근로기준법, 임금, 퇴직금, 해고, 산재, 주휴수당 등
   - "형사법": 범죄, 형벌, 수사, 재판, 고소/고발 등
   - "민사법": 계약, 손해배상, 소유권, 채권 등
   - "기타": 위 카테고리에 속하지 않는 법률 질문

2. needs_clarification: 질문이 극도로 모호하여 어떤 답변도 불가능한지 (true/false)
3. needs_case_law: 대법원 판례가 필요한지 (true/false)
4. clarification_question: needs_clarification이 true일 때만 작성""",
        ),
        ("human", "{query}"),
    ])

    def analyze_query(state: AgentState) -> AgentState:
        query = state["user_query"]
        print("🔎 [질문 분석 중...]")
        chain = analyze_prompt | structured_llm
        analysis: QueryAnalysis = chain.invoke({"query": query})
        print(f"📋 [분석 결과] 분야: {analysis.category} / 판례 필요: {'예' if analysis.needs_case_law else '아니오'}")
        return {"query_analysis": analysis.model_dump()}

    return analyze_query


def create_clarify_node(llm: ChatOpenAI):
    def request_clarification(state: AgentState) -> AgentState:
        analysis = state.get("query_analysis", {})
        clarification_q = analysis.get("clarification_question", "") or "질문을 좀 더 구체적으로 알려주세요. 상황, 상대방, 쟁점, 원하는 결과를 적어주시면 더 정확히 답변드릴 수 있습니다."
        print("❓ [명확화 요청]")
        answer = f"""안녕하세요! 질문을 더 이해하기 위해 몇 가지를 확인하고 싶어요.

{clarification_q}

위 내용을 포함해 다시 알려주시면 더 정확히 도움 드릴 수 있습니다. 😊"""
        return {"generated_answer": answer, "next_action": "end"}

    return request_clarification


# ----------------------------
# 검색 관련 헬퍼
# ----------------------------
NON_STATUTE_SOURCES = {"interpretation", "case_law", "moel_qa", "판정선례"}


def expand_queries(query: str) -> List[str]:
    variants = {query.strip()}
    # 조사/불용어 일부 제거 시도
    compact = re.sub(r"[\s]+", " ", query).strip()
    variants.add(compact)
    # 괄호/슬래시 제거 버전
    variants.add(re.sub(r"[()\[\]/]", " ", compact))
    # 영어 질문 대응: 한국어 번역 힌트가 없다면 그대로 사용
    return [v for v in variants if v]


def dedup_documents(docs: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for doc in docs:
        key = doc.metadata.get("id") or (doc.metadata.get("source"), doc.metadata.get("law_name"), doc.metadata.get("article_no"), doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def boost_non_statute_score(doc: Document, boost: float = 0.15) -> float:
    score = doc.metadata.get("relevance_score", 0.0)
    if str(doc.metadata.get("source", "")) in NON_STATUTE_SOURCES:
        score += boost
    return score


def format_context_snippets(docs: List[Document], max_docs: int = 5, max_chars: int = 500) -> str:
    parts = []
    for i, doc in enumerate(docs[:max_docs], 1):
        meta = doc.metadata or {}
        law_name = meta.get("law_name", "")
        article = meta.get("article_no", "")
        title = meta.get("article_title") or meta.get("title", "")
        source = meta.get("source", "")
        snippet = doc.page_content[: max_chars].strip()
        header = f"[근거 {i}]"
        if law_name:
            header += f" {law_name}"
            if article:
                header += f" 제{article}조"
        if title:
            header += f" - {title}"
        if source and not law_name:
            header += f" ({source})"
        parts.append(f"{header}\n{snippet}\n")
    return "\n".join(parts) if parts else "(관련 법령 문서가 검색되지 않았습니다)"


def create_search_node(vectorstore: QdrantVectorStore):
    def search_documents(state: AgentState) -> AgentState:
        query = state["user_query"]
        print(f"🔍 [법령 검색] 쿼리: {query[:50]}...")

        variants = expand_queries(query)
        all_docs: List[Document] = []
        for q in variants:
            try:
                res = vectorstore.similarity_search_with_score(q, k=12)
                all_docs.extend([doc for doc, score in res])
            except Exception as e:
                print(f"⚠️  [검색 오류] {e}")

        all_docs = dedup_documents(all_docs)
        if not all_docs:
            print("⚠️  [검색 결과 없음]")
            return {"retrieved_docs": []}

        try:
            reranker = JinaReranker(top_n=6)
            reranked = reranker.compress_documents(all_docs, query)
            if reranked:
                # 노동법 비법령 문서 가중치 부여 후 재정렬
                reranked = sorted(
                    reranked,
                    key=lambda d: boost_non_statute_score(d),
                    reverse=True,
                )
                docs = reranked[:6]
                print(f"✅ [리랭킹 완료] {len(docs)}개 문서 선별")
            else:
                docs = all_docs[:6]
                print("⚠️  [리랭킹 결과 없음] 원본 상위 6개 사용")
        except Exception as e:
            print(f"⚠️  [리랭킹 오류] {e}")
            docs = all_docs[:6]

        for i, d in enumerate(docs, 1):
            print(f"   [{i}] score={d.metadata.get('relevance_score', 0):.4f} | {d.page_content[:40]}...")

        return {"retrieved_docs": docs}

    return search_documents


def create_case_law_search_node(llm: ChatOpenAI):
    def search_case_law(state: AgentState) -> AgentState:
        query = state["user_query"]
        analysis = state.get("query_analysis", {})
        category = analysis.get("category", "기타")

        print("⚖️  [판례 검색] 대법원 판례 웹 검색 중...")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print("⚠️  [판례 검색 스킵] TAVILY_API_KEY 미설정")
            return {"case_law_results": []}

        try:
            search_tool = TavilySearchResults(
                max_results=3,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
            )
            search_query = f"대법원 판례 {category} {query}"
            results = search_tool.invoke({"query": search_query})
            case_laws = []
            for r in results:
                case_laws.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:400],
                    }
                )
            print(f"✅ [판례 검색 완료] {len(case_laws)}건")
            return {"case_law_results": case_laws}
        except Exception as e:
            print(f"⚠️  [판례 검색 오류] {e}")
            return {"case_law_results": []}

    return search_case_law


def create_generate_node(llm: ChatOpenAI):
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 법률 전문 AI 'A-TEAM 봇'입니다.
- 검색된 근거를 인용하여 답변합니다.
- 답변 구조: 📌 결론 → 📖 법적 근거 → 💡 추가 설명
- 근거마다 [법령명 제N조], [판례: 제목] 형태로 표기하고, 존재하는 근거만 사용합니다.
- 불확실하면 추측하지 말고 한계를 명시합니다.
- 한국어로 간결하게 답변합니다.""",
        ),
        (
            "human",
            """질문 분야: {category}
사용자 질문: {query}

📚 근거 스니펫:
{context}

⚖️ 관련 판례:
{case_law}

위 근거를 인용해 답변하세요. 각 단락에 근거를 붙이고, 근거가 없으면 모른다고 말하세요.""",
        ),
    ])

    def generate_answer(state: AgentState) -> AgentState:
        query = state["user_query"]
        analysis = state.get("query_analysis", {})
        category = analysis.get("category", "기타")
        docs = state.get("retrieved_docs", []) or []
        case_laws = state.get("case_law_results", []) or []

        print("💬 [답변 생성 중...]")

        context = format_context_snippets(docs, max_docs=5, max_chars=500)

        if case_laws:
            case_parts = []
            for i, case in enumerate(case_laws, 1):
                case_parts.append(f"[판례 {i}] {case.get('title','')}: {case.get('content','')}")
            case_law_context = "\n".join(case_parts)
        else:
            case_law_context = "(관련 판례 정보 없음)"

        if not docs and not case_laws:
            answer = """죄송합니다. 관련 근거를 찾지 못했습니다. 질문을 더 구체적으로 작성하거나 다른 키워드로 다시 시도해 주세요. 복잡한 사안이면 전문 법률 상담을 권장드립니다."""
        else:
            chain = answer_prompt | llm
            response = chain.invoke(
                {
                    "category": category,
                    "query": query,
                    "context": context,
                    "case_law": case_law_context,
                }
            )
            answer = response.content

        print("✅ [답변 생성 완료]")
        return {"generated_answer": answer}

    return generate_answer


def route_after_analysis(state: AgentState) -> Literal["clarify", "search"]:
    analysis = state.get("query_analysis", {})
    if analysis.get("needs_clarification", False):
        return "clarify"
    return "search"


def route_after_search(state: AgentState) -> Literal["case_law_search", "generate"]:
    analysis = state.get("query_analysis", {})
    if analysis.get("needs_case_law", False):
        return "case_law_search"
    return "generate"


def initialize_resources():
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY가 .env 파일에 설정되지 않았습니다!")

    print("🔧 설정 로드 완료")
    print("\n🚀 임베딩 모델 로드 중 (Qwen/Qwen3-Embedding-0.6B)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ 임베딩 모델 로드 완료")

    print("\n📡 Qdrant 연결 중...")
    warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30, prefer_grpc=False)
    print("✅ Qdrant 연결 완료")

    print("\n🗂️  벡터스토어 초기화 중...")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="text",
    )
    print("✅ 벡터스토어 초기화 완료")
    return {"embeddings": embeddings, "vectorstore": vectorstore}


def initialize_langgraph_chatbot():
    resources = initialize_resources()
    vectorstore = resources["vectorstore"]

    print("\n🤖 LLM 설정 중...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    print("✅ LLM 설정 완료")

    print("\n⚙️  LangGraph 노드 생성 중...")
    analyze_node = create_analyze_query_node(llm)
    clarify_node = create_clarify_node(llm)
    search_node = create_search_node(vectorstore)
    case_law_node = create_case_law_search_node(llm)
    generate_node = create_generate_node(llm)
    print("✅ 노드 생성 완료")

    workflow = StateGraph(AgentState)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("search", search_node)
    workflow.add_node("case_law_search", case_law_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges("analyze", route_after_analysis, {"clarify": "clarify", "search": "search"})
    workflow.add_edge("clarify", END)
    workflow.add_conditional_edges("search", route_after_search, {"case_law_search": "case_law_search", "generate": "generate"})
    workflow.add_edge("case_law_search", "generate")
    workflow.add_edge("generate", END)

    graph = workflow.compile()
    print("✅ LangGraph 구성 완료")
    return graph


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  경고: TAVILY_API_KEY가 설정되지 않았습니다. 판례 검색이 비활성화됩니다.\n")

    try:
        print("\n" + "=" * 60)
        print("🚀 A-TEAM 법률 RAG 챗봇 (LangGraph V5) 초기화")
        print("=" * 60 + "\n")

        graph = initialize_langgraph_chatbot()

        print("\n" + "=" * 60)
        print("✅ 🤖 A-TEAM 법률 챗봇 준비 완료 (V5)")
        print("=" * 60)
        print("\n사용 방법: 노동법/형사법/민사법 질문에 답변, 판례 필요 시 웹 검색, 모호하면 명확화 요청")
        print("'exit', 'quit', '종료'로 종료합니다.\n")

        while True:
            try:
                user_input = input("👤 User >> ").strip()
                if user_input.lower() in ["exit", "quit", "종료", "q"]:
                    print("\n👋 챗봇을 종료합니다. 감사합니다!")
                    break
                if not user_input:
                    print("❌ 질문을 입력해주세요.\n")
                    continue

                initial_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_query": user_input,
                    "query_analysis": None,
                    "retrieved_docs": None,
                    "case_law_results": None,
                    "generated_answer": None,
                    "next_action": None,
                }

                print("\n" + "-" * 60)
                print("🔄 워크플로우 실행 중...")
                print("-" * 60 + "\n")

                result = graph.invoke(initial_state)
                answer = result.get("generated_answer", "")
                if answer:
                    print("\n" + "=" * 60)
                    print("🤖 AI 답변:")
                    print("=" * 60)
                    print(f"\n{answer}\n")
                    print("=" * 60 + "\n")
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
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
