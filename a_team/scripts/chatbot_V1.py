import os
import warnings
from pathlib import Path
from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv

# Qdrant & LangChain 관련 임포트
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# 환경 변수 로드: 실행 위치(CWD)와 무관하게 이 파일과 같은 폴더의 .env를 사용
_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH)


# ===========================
# State 정의
# ===========================
class AgentState(TypedDict):
    """LangGraph Agent의 상태를 정의하는 TypedDict"""
    # messages: 대화 히스토리 (자동으로 추가됨)
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ===========================
# Tool 정의
# ===========================
def create_legal_search_tool(vectorstore: QdrantVectorStore):
    """법률 검색 Tool 생성"""
    
    @tool("legal_search_tool")
    def legal_search_tool(query: str) -> str:
        """법률/판례/행정해석을 Qdrant에서 검색해 관련 문서를 반환합니다."""
        
        k = 5  # 검색 결과 개수
        max_chars = 1200  # 문서당 최대 문자 수
        
        results = vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return "검색 결과가 없습니다. 질문을 더 구체적으로 입력해 주세요."
        
        lines = []
        for i, (doc, score) in enumerate(results, start=1):
            # 메타데이터 추출
            metadata = doc.metadata
            source = metadata.get("source", "unknown")
            title = metadata.get("title", "")
            chunk_info = f"청크 {metadata.get('chunk_index', 0)+1}/{metadata.get('total_chunks', 1)}"
            
            lines.append(f"[문서 {i}] score={score:.4f} | {source} | {title} | {chunk_info}")
            
            content = (doc.page_content or "").strip()
            if content:
                if max_chars > 0 and len(content) > max_chars:
                    content = content[:max_chars].rstrip() + "…"
                lines.append(content)
            else:
                lines.append("(본문 없음)")
            lines.append("")
        
        return "\n".join(lines).strip()
    
    return legal_search_tool


# ===========================
# 그래프 노드 함수 정의
# ===========================
def create_agent_node(llm_with_tools):
    """Agent 노드: LLM이 사용자 질문에 응답하거나 Tool을 호출"""
    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    return agent_node


def should_continue(state: AgentState) -> str:
    """조건부 엣지: Tool 호출이 있으면 'tools'로, 없으면 'end'로 라우팅"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Tool calls가 있으면 tools 노드로 이동
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 없으면 종료
    return "end"


# ===========================
# LangGraph 초기화
# ===========================
def initialize_langgraph_chatbot():
    """LangGraph 기반 RAG 챗봇 초기화"""
    
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
    
    # 5. Tool 생성
    print(f"\n🛠️  Tool 생성 중...")
    legal_tool = create_legal_search_tool(vectorstore)
    tools = [legal_tool]
    print("✅ Tool 생성 완료")
    
    # 6. LLM 설정
    print(f"\n🤖 LLM 설정 중...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )
    
    # Tool을 LLM에 바인딩
    llm_with_tools = llm.bind_tools(tools)
    print("✅ LLM 설정 완료")
    
    # 7. 시스템 프롬프트 생성
    system_prompt = """당신은 법률 전문 AI 어시스턴트 'A-TEAM 봇'입니다.

역할:
- 사용자의 법률 관련 질문에 정확하고 친절하게 답변합니다.
- legal_search_tool을 사용하여 관련 법률 정보를 검색합니다.
- 검색된 법령, 판례, 행정해석을 바탕으로 근거 있는 답변을 제공합니다.

답변 원칙:
1. 검색된 자료를 바탕으로 답변하세요.
2. 법령명, 조항, 판례번호 등 구체적인 근거를 제시하세요.
3. 법률 용어는 쉽게 풀어서 설명하세요.
4. 확실하지 않은 내용은 추측하지 말고, 검색 도구를 활용하세요.
5. 한국어로 답변하세요."""
    
    # 8. LangGraph 생성
    print(f"\n⚙️  LangGraph 구성 중...")
    
    # StateGraph 초기화
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    agent_node_func = create_agent_node(llm_with_tools)
    workflow.add_node("agent", agent_node_func)
    workflow.add_node("tools", ToolNode(tools))
    
    # 엣지 추가
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # 그래프 컴파일
    graph = workflow.compile()
    print("✅ LangGraph 구성 완료")
    
    return graph, system_prompt


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
        print("🚀 A-TEAM 법률 RAG 챗봇 (LangGraph) 초기화 시작")
        print("="*60 + "\n")
        
        graph, system_prompt = initialize_langgraph_chatbot()
        
        print("\n" + "="*60)
        print("✅ 🤖 A-TEAM 법률 챗봇 준비 완료!")
        print("="*60)
        print("\n💡 사용 방법:")
        print("  - 노동분야 법률, 형사법, 민사법 관련 질문에 응답할 수 있습니다.")
        print("  - 'exit', 'quit', '종료'를 입력하면 종료됩니다")
        print("="*60 + "\n")
        
        # 대화 기록 저장 (메시지 리스트로 관리)
        messages = []
        
        # 시스템 메시지 추가
        from langchain_core.messages import SystemMessage
        messages.append(SystemMessage(content=system_prompt))
        
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
                
                # 사용자 메시지 추가
                messages.append(HumanMessage(content=user_input))
                
                # 그래프 실행
                print()  # 줄바꿈
                result = graph.invoke({"messages": messages})
                
                # 최종 응답 추출
                response_messages = result["messages"]
                ai_response = None
                
                # 마지막 AIMessage 찾기
                for msg in reversed(response_messages):
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        ai_response = msg.content
                        break
                
                if ai_response:
                    print(f"\n🤖 AI >> {ai_response}\n")
                    print("-" * 60 + "\n")
                    # 전체 메시지 히스토리 업데이트
                    messages = response_messages
                else:
                    print("\n⚠️ 응답을 생성할 수 없습니다.\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다. 감사합니다!")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("💡 다시 시도해주세요.\n")
    
    except Exception as e:
        print(f"\n❌ 챗봇 초기화 실패: {e}")
        print("💡 설정을 확인하고 다시 시도해주세요.")
        raise


if __name__ == "__main__":
    main()
