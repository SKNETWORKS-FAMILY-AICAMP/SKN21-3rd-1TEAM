import os
import warnings
from pathlib import Path
from typing import Annotated, TypedDict, Sequence, Optional, List
from dotenv import load_dotenv

# Qdrant & LangChain 관련 임포트
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

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
    # 질문 분류 결과
    query_classification: Optional[str]
    # 검색 결과 (Document 리스트)
    retrieved_docs: Optional[List[Document]]
    # 생성된 답변
    generated_answer: Optional[str]
    # 검증 결과
    validation_result: Optional[bool]
    # 검증 피드백
    validation_feedback: Optional[str]
    # 재시도 횟수
    retry_count: int


# ===========================
# 검색 함수 정의 (사전 준비 영역)
# ===========================
def create_search_function(vectorstore: QdrantVectorStore):
    """법률 검색 함수 생성"""
    
    def search_legal_docs(query: str, k: int = 5) -> List[tuple]:
        """
        법률/판례/행정해석을 Qdrant에서 검색
        
        Args:
            query: 검색 쿼리
            k: 검색 결과 개수
            
        Returns:
            (Document, score) 튜플의 리스트
        """
        results = vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    return search_legal_docs


# ===========================
# 노드 함수 정의 (LangGraph 영역)
# ===========================

def create_classify_node(llm: ChatOpenAI):
    """노드 1: 질문 분류"""
    
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 질문을 분류하는 전문가입니다.
사용자의 질문을 다음 카테고리 중 하나로 분류하세요:

1. 노동법 - 근로기준법, 노동조합, 임금, 퇴직금, 해고 등
2. 형사법 - 범죄, 형벌, 수사, 재판 등
3. 민사법 - 계약, 손해배상, 소유권, 채권 등
4. 기타 - 위 카테고리에 속하지 않는 법률 질문

분류 결과만 반환하세요. 예: "노동법", "형사법", "민사법", "기타" """),
        ("human", "{query}")
    ])
    
    def classify_query(state: AgentState) -> AgentState:
        """질문 분류 노드"""
        query = state["user_query"]
        
        chain = classify_prompt | llm
        response = chain.invoke({"query": query})
        classification = response.content.strip()
        
        print(f"📋 [질문 분류] {classification}")
        
        return {
            "query_classification": classification
        }
    
    return classify_query


def create_search_node(search_function):
    """노드 2: 검색 실행"""
    
    def search_documents(state: AgentState) -> AgentState:
        """검색 실행 노드"""
        query = state["user_query"]
        classification = state.get("query_classification", "기타")
        
        # 검색 수행
        print(f"🔍 [검색 실행] 쿼리: {query[:50]}...")
        
        # 재시도 시 검색 개수 증가
        retry_count = state.get("retry_count", 0)
        k = 5 + (retry_count * 3)  # 재시도마다 3개씩 더 검색
        
        results = search_function(query, k=k)
        
        if results:
            docs = [doc for doc, score in results]
            print(f"✅ [검색 완료] {len(docs)}개 문서 검색됨")
        else:
            docs = []
            print(f"⚠️  [검색 결과 없음]")
        
        return {
            "retrieved_docs": docs
        }
    
    return search_documents


def create_generate_node(llm: ChatOpenAI):
    """노드 3: 답변 생성"""
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 전문 AI 어시스턴트 'A-TEAM 봇'입니다.

역할:
- 검색된 법률 문서를 바탕으로 정확하고 친절하게 답변합니다.
- 법령명, 조항, 판례번호 등 구체적인 근거를 제시합니다.
- 법률 용어는 쉽게 풀어서 설명합니다.

답변 작성 시:
1. 검색된 자료만을 근거로 답변하세요.
2. 답변은 구조화하여 작성하세요 (결론 → 근거 → 추가 설명).
3. 관련 법령과 조항을 명시하세요.
4. 확실하지 않은 내용은 추측하지 마세요.
5. 한국어로 답변하세요."""),
        ("human", """질문 카테고리: {classification}

사용자 질문: {query}

검색된 관련 문서:
{context}

위 자료를 바탕으로 질문에 답변해주세요.""")
    ])
    
    def generate_answer(state: AgentState) -> AgentState:
        """답변 생성 노드"""
        query = state["user_query"]
        classification = state.get("query_classification", "기타")
        docs = state.get("retrieved_docs", [])
        
        print(f"💬 [답변 생성 중...]")
        
        if not docs:
            answer = "죄송합니다. 관련된 법률 정보를 찾을 수 없습니다. 질문을 더 구체적으로 작성해주시거나, 다른 방식으로 질문해주세요."
        else:
            # 문서를 컨텍스트로 포맷팅
            context_parts = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                source = metadata.get("source", "unknown")
                title = metadata.get("title", "")
                content = doc.page_content[:1000]  # 문서당 최대 1000자
                
                context_parts.append(f"[문서 {i}] {source} - {title}\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # LLM으로 답변 생성
            chain = answer_prompt | llm
            response = chain.invoke({
                "classification": classification,
                "query": query,
                "context": context
            })
            answer = response.content
        
        print(f"✅ [답변 생성 완료]")
        
        return {
            "generated_answer": answer
        }
    
    return generate_answer


def create_validation_node(llm: ChatOpenAI):
    """노드 4: 검증"""
    
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 답변의 품질을 검증하는 전문가입니다.

답변을 평가하여 다음 기준을 확인하세요:
1. 검색된 문서를 근거로 답변했는가?
2. 법령명이나 조항 등 구체적인 근거가 있는가?
3. 질문에 직접적으로 답변했는가?
4. 답변이 충분히 상세한가?

검증 결과를 JSON 형식으로 반환하세요:
{
  "valid": true/false,
  "feedback": "검증 결과에 대한 피드백"
}"""),
        ("human", """사용자 질문: {query}

검색된 문서 개수: {doc_count}

생성된 답변:
{answer}

위 답변을 검증해주세요.""")
    ])
    
    def validate_answer(state: AgentState) -> AgentState:
        """답변 검증 노드"""
        query = state["user_query"]
        answer = state.get("generated_answer", "")
        docs = state.get("retrieved_docs", [])
        
        print(f"🔍 [답변 검증 중...]")
        
        # 기본 검증: 답변이 너무 짧거나 검색 결과가 없으면 실패
        if len(answer) < 50 or not docs:
            print(f"❌ [검증 실패] 답변이 불충분합니다.")
            return {
                "validation_result": False,
                "validation_feedback": "답변이 너무 짧거나 검색 결과가 없습니다."
            }
        
        # LLM으로 검증
        chain = validation_prompt | llm
        response = chain.invoke({
            "query": query,
            "doc_count": len(docs),
            "answer": answer
        })
        
        # 응답 파싱 (간단하게 "valid": true/false 찾기)
        content = response.content.lower()
        is_valid = "true" in content or "통과" in content or "적절" in content
        
        if is_valid:
            print(f"✅ [검증 통과]")
        else:
            print(f"⚠️  [검증 실패] 재시도가 필요할 수 있습니다.")
        
        return {
            "validation_result": is_valid,
            "validation_feedback": response.content
        }
    
    return validate_answer


def create_retry_decision_node():
    """노드 5: 재시도 판단"""
    
    def decide_retry(state: AgentState) -> AgentState:
        """재시도 판단 노드"""
        retry_count = state.get("retry_count", 0)
        validation_result = state.get("validation_result", False)
        
        if not validation_result and retry_count < 2:  # 최대 2번 재시도
            new_retry_count = retry_count + 1
            print(f"🔄 [재시도 {new_retry_count}/2] 검색을 다시 시도합니다.")
            return {"retry_count": new_retry_count}
        elif not validation_result:
            print(f"⚠️  [재시도 제한 도달] 현재 답변을 반환합니다.")
        
        return {"retry_count": retry_count}
    
    return decide_retry


# ===========================
# 조건부 엣지 함수
# ===========================
def should_retry(state: AgentState) -> str:
    """검증 후 재시도 여부 결정"""
    validation_result = state.get("validation_result", False)
    retry_count = state.get("retry_count", 0)
    
    if not validation_result and retry_count < 2:
        return "retry"  # 재시도 노드로
    else:
        return "end"  # 종료


# ===========================
# 사전 준비 영역: 리소스 초기화
# ===========================
def initialize_resources():
    """임베딩 모델, 벡터스토어, Retriever, Tools 초기화"""
    
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
    
    # 5. Retriever 생성 (사용 안 할 수도 있지만 준비)
    print(f"\n🔍 Retriever 생성 중...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    print("✅ Retriever 생성 완료")
    
    # 6. 검색 함수 생성
    print(f"\n🛠️  검색 함수 생성 중...")
    search_function = create_search_function(vectorstore)
    print("✅ 검색 함수 생성 완료")
    
    return {
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "search_function": search_function
    }


# ===========================
# LangGraph 초기화
# ===========================
def initialize_langgraph_chatbot():
    """LangGraph 기반 RAG 챗봇 초기화"""
    
    # 사전 준비: 리소스 초기화
    resources = initialize_resources()
    search_function = resources["search_function"]
    
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
    classify_node = create_classify_node(llm)
    search_node = create_search_node(search_function)
    generate_node = create_generate_node(llm)
    validation_node = create_validation_node(llm)
    retry_node = create_retry_decision_node()
    print("✅ 노드 생성 완료")
    
    # StateGraph 구성
    print(f"\n🔗 LangGraph 워크플로우 구성 중...")
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("classify", classify_node)
    workflow.add_node("search", search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("retry_decision", retry_node)
    
    # 엣지 추가
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "search")
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", "validate")
    
    # 조건부 엣지: 검증 후 재시도 또는 종료
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "retry": "retry_decision",
            "end": END
        }
    )
    workflow.add_edge("retry_decision", "search")
    
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
        print("  - 노동분야 법률, 형사법, 민사법 관련 질문에 응답할 수 있습니다.")
        print("  - 'exit', 'quit', '종료'를 입력하면 종료됩니다")
        print("\n📊 워크플로우:")
        print("  1. 질문 분류 → 2. 검색 실행 → 3. 답변 생성")
        print("  → 4. 검증 → 5. 재시도 판단 (필요시)")
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
                    "query_classification": None,
                    "retrieved_docs": None,
                    "generated_answer": None,
                    "validation_result": None,
                    "validation_feedback": None,
                    "retry_count": 0
                }
                
                # 그래프 실행
                print("\n" + "="*60)
                print("🔄 워크플로우 실행 중...")
                print("="*60 + "\n")
                
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
