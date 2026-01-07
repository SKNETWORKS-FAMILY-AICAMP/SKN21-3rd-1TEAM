import os
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# 환경 변수 로드
_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH)

def initialize_rag_chatbot():
    """로컬 Qdrant 기반 RAG 챗봇 초기화"""
    
    # 1. 설정
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "legal_data_local")
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    
    print(f"🔧 설정 로드 완료")
    print(f"  - Qdrant: {QDRANT_HOST}:{QDRANT_PORT} (로컬)")
    print(f"  - Collection: {COLLECTION_NAME}")
    
    # 2. 임베딩 모델 설정
    print(f"\n🚀 임베딩 모델 로드 중 (Qwen/Qwen3-Embedding-0.6B)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'trust_remote_code': True}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로드 완료")
    
    # 3. 로컬 Qdrant 클라이언트 연결
    print(f"\n📡 로컬 Qdrant 연결 중...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # 컬렉션 존재 확인 및 생성
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"⚠️  컬렉션 '{COLLECTION_NAME}'이 없습니다. 생성 중...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=896, distance=Distance.COSINE)  # Qwen3-Embedding 차원
        )
        print(f"✅ 컬렉션 '{COLLECTION_NAME}' 생성 완료")
        print("💡 데이터를 업로드해야 검색이 가능합니다.")
    else:
        print(f"✅ 로컬 Qdrant 연결 완료")
    
    # 4. 벡터스토어 생성
    print(f"\n🗂️  벡터스토어 초기화 중...")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="text"
    )
    print("✅ 벡터스토어 초기화 완료")
    
    # 5. Retriever 생성
    print(f"\n🔍 Retriever 생성 중...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    print("✅ Retriever 생성 완료")
    
    # 6. Tool 생성
    print(f"\n🛠️  Tool 생성 중...")
    
    @tool("legal_search_tool")
    def legal_search_tool(query: str) -> str:
        """법률 정보를 Qdrant에서 검색해 관련 문서를 반환합니다."""
        
        k = int(os.getenv("RETRIEVAL_K", "5"))
        max_chars = int(os.getenv("RETRIEVAL_DOC_CHARS", "1200"))
        
        results = vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return "검색 결과가 없습니다. 질문을 더 구체적으로 입력해 주세요."
        
        lines = []
        for i, (doc, score) in enumerate(results, start=1):
            doc_id = doc.metadata.get("_id", "")
            lines.append(f"[문서 {i}] score={score:.4f} id={doc_id}")
            
            content = (doc.page_content or "").strip()
            if content:
                if max_chars > 0 and len(content) > max_chars:
                    content = content[:max_chars].rstrip() + "…"
                lines.append(content)
            else:
                lines.append("(본문 없음)")
            lines.append("")
        
        return "\n".join(lines).strip()
    
    tools = [legal_search_tool]
    print("✅ Tool 생성 완료")
    
    # 7. LLM 설정
    print(f"\n🤖 LLM 설정 중...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )
    print("✅ LLM 설정 완료")
    
    # 8. 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 전문 AI 어시스턴트 'A-TEAM 봇'입니다.

역할:
- 사용자의 법률 관련 질문에 정확하고 친절하게 답변합니다.
- legal_search_tool을 사용하여 관련 법률 정보를 검색합니다.
- 검색된 법령, 판례, 행정해석을 바탕으로 근거 있는 답변을 제공합니다.

답변 원칙:
1. 검색된 자료를 바탕으로 답변하세요.
2. 법령명, 조항, 판례번호 등 구체적인 근거를 제시하세요.
3. 법률 용어는 쉽게 풀어서 설명하세요.
4. 확실하지 않은 내용은 추측하지 말고, 검색 도구를 활용하세요.
5. 한국어로 답변하세요."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 9. Agent 생성
    print(f"\n⚙️  Agent 생성 중...")
    agent = create_tool_calling_agent(llm, tools, prompt)
    print("✅ Agent 생성 완료")
    
    # 10. AgentExecutor 생성
    print(f"\n🎯 AgentExecutor 생성 중...")
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=False
    )
    print("✅ AgentExecutor 생성 완료")
    
    return agent_executor


def main():
    """RAG 챗봇 실행 메인 함수"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("💡 .env 파일에 OPENAI_API_KEY를 추가하세요.")
        return
    
    try:
        print("\n" + "="*60)
        print("🚀 A-TEAM 법률 RAG 챗봇 초기화 시작 (로컬 Qdrant)")
        print("="*60 + "\n")
        
        chatbot = initialize_rag_chatbot()
        
        print("\n" + "="*60)
        print("✅ 🤖 A-TEAM 법률 챗봇 준비 완료!")
        print("="*60)
        print("\n💡 사용 방법:")
        print("  - 법률 관련 질문을 입력하세요")
        print("  - 'exit', 'quit', '종료'를 입력하면 종료됩니다")
        print("="*60 + "\n")
        
        chat_history = []
        
        while True:
            try:
                user_input = input("👤 User >> ").strip()
                
                if user_input.lower() in ["exit", "quit", "종료", "q"]:
                    print("\n👋 챗봇을 종료합니다. 감사합니다!")
                    break
                
                if not user_input:
                    print("❌ 질문을 입력해주세요.\n")
                    continue
                
                print()
                response = chatbot.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                print(f"\n🤖 AI >> {response['output']}\n")
                print("-" * 60 + "\n")
                
                chat_history.append(("human", user_input))
                chat_history.append(("ai", response["output"]))
                
            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다. 감사합니다!")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("💡 다시 시도해주세요.\n")
                
    except Exception as e:
        print(f"\n❌ 챗봇 초기화 실패: {e}")
        print("💡 설정을 확인하고 다시 시도해주세요.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
