################################################
# A-TEAM 법률 RAG 챗봇 (LangChain V3)
  # 벡터 검색 + BM25 하이브리드 검색
  # Jina Reranker 기반 문서 리랭킹
  # LangGraph 제거 -> 순수 LangChain 및 절차적 로직으로 변경
# 작성자 정보
  # 작성자: SKN 3-1팀 A-TEAM
  # 작성일: 2026-01-08
################################################

import os
import re
import warnings
from pathlib import Path
from typing import Optional, List, Any, Sequence
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from a_team.scripts.bm25_search import BM25KeywordRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document, BaseDocumentCompressor
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH)


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


# ----------------------------
# 검색 관련 헬퍼
# ----------------------------
def expand_queries(query: str) -> List[str]:
    variants = {query.strip()}
    # 조사/불용어 일부 제거 시도
    compact = re.sub(r"[\s]+", " ", query).strip()
    variants.add(compact)
    # 괄호/슬래시 제거 버전
    variants.add(re.sub(r"[()\[\]/]", " ", compact))
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


def retrieve_documents(query: str, vectorstore: QdrantVectorStore, bm25_retriever: Optional[BM25KeywordRetriever]) -> List[Document]:
    print(f"🔍 [법령 검색] 쿼리: {query[:50]}...")

    variants = expand_queries(query)
    all_docs: List[Document] = []
    vector_scores = []
    
    # 1. 벡터 검색 (cosine similarity)
    for q in variants:
        try:
            res = vectorstore.similarity_search_with_score(q, k=10)
            all_docs.extend([doc for doc, score in res])
            vector_scores.extend([score for doc, score in res])
        except Exception as e:
            print(f"⚠️  [벡터 검색 오류] {e}")

    # 벡터 검색 결과가 있고, 모든 score가 0.5 이하라면 쿼리 변형 후 재검색 시도
    if vector_scores and all(s <= 0.5 for s in vector_scores):
        print("⚠️  [벡터 유사도 0.5 이하, 쿼리 변형 후 재검색]")
        import re
        keywords = re.findall(r"[\w가-힣]+", query)
        simple_query = " ".join(keywords)
        retry_variants = expand_queries(simple_query)
        all_docs = []
        vector_scores = []
        for q in retry_variants:
            try:
                res = vectorstore.similarity_search_with_score(q, k=10)
                all_docs.extend([doc for doc, score in res])
                vector_scores.extend([score for doc, score in res])
            except Exception as e:
                print(f"⚠️  [벡터 재검색 오류] {e}")
        
        # BM25도 재검색
        if bm25_retriever:
            for q in retry_variants:
                try:
                    bm25_docs = bm25_retriever.search(q, k=5)
                    all_docs.extend(bm25_docs)
                except Exception as e:
                    print(f"⚠️  [BM25 재검색 오류] {e}")
    else:
        # 2. BM25/keyword 검색 (하이브리드)
        if bm25_retriever:
            for q in variants:
                try:
                    bm25_docs = bm25_retriever.search(q, k=5)
                    all_docs.extend(bm25_docs)
                except Exception as e:
                    print(f"⚠️  [BM25 검색 오류] {e}")

    all_docs = dedup_documents(all_docs)
    if not all_docs:
        print("⚠️  [검색 결과 없음]")
        return []

    try:
        reranker = JinaReranker(top_n=6)
        reranked = reranker.compress_documents(all_docs, query)
        if reranked:
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

    return docs


def generate_answer(query: str, docs: List[Document], llm: ChatOpenAI) -> str:
    print("💬 [답변 생성 중...]")

    context = format_context_snippets(docs, max_docs=5, max_chars=500)
    
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 대한민국 노동 분야 법률, 형사법 법률, 민사법 법률에 대해 전문적으로 학습된 AI 도우미입니다.
            사용자의 질문에 대해 저장된 법률 조항 데이터와 관련 정보(판례, 행정해석 등)를 기반으로 정확하고 신뢰성 있는 답변을 제공하세요.
            1. 답변 작성 기본 지침 
                - 법률 조항에 관한 질문이라면 그 조항에 관한 전체 내용을 가져온다.
                - 예를들어 '근로기준법 제1조의 내용'이라는 질문을 받으면 근로기준법 제1조의 조항을 전부 다 답변한다.
                - 질문 유형에 따라 관련 정보를 구조적으로 작성하며, 중요 세법 조문과 요약된 내용을 포함합니다.
                - 비전문가도 이해할 수 있도록 용어를 친절히 설명합니다.
            2. 답변 작성 세부 지침:
                - **간결성**: 답변은 간단하고 명확하게 작성하되, 법 조항에 관한 질문일 경우 관련 법 조문의 전문을 명시합니다.
                - **구조화된 정보 제공**:
                    - 세법 조항 번호, 세법 조항의 정의, 시행령, 관련 규정을 구체적으로 명시합니다.
                    - 복잡한 개념은 예시를 들어 설명하거나, 단계적으로 안내합니다.
                - **신뢰성 강조**:
                    - 답변이 법적 조언이 아니라 정보 제공 목적임을 명확히 알립니다.
                    - "이 답변은 세법 관련 정보를 바탕으로 작성되었으며, 구체적인 상황에 따라 전문가의 추가 조언이 필요할 수 있습니다."를 추가합니다.
                - **정확성**:
                    - 법령 및 법률에 관한질문은 추가적인 내용없이 한가지 content에 집중하여 답변한다.
                    - 조항에 대한 질문은 시행령이나 시행규칙보단 해당법에서 가져오는것에 집중한다.
            3. 추가적인 사용자 지원:
                - 답변 후 사용자에게 주제와 관련된 후속 질문 두 가지를 제안합니다.
                - 후속 질문은 사용자가 더 깊이 탐구할 수 있도록 설계하며, 각 질문 앞뒤에 한 줄씩 띄어쓰기를 합니다.

            4. 예외 상황 처리:
                - 사용자가 질문을 모호하게 작성한 경우:
                    - "질문이 명확하지 않습니다. 구체적으로 어떤 부분을 알고 싶으신지 말씀해 주시겠어요?"와 같은 문구로 추가 정보를 요청합니다.
                - 질문이 알고 있는 법률(노동 분야 법률, 형사법, 민사법)과 직접 관련이 없는 경우:
                    - "이 질문은 제가 학습한 법률 범위를 벗어납니다."라고 알리고, 알고 있는 법률(노동 분야 법률, 형사법, 민사법)과 관련된 새로운 질문을 유도합니다.

            5. 추가 지침:
                - 개행문자 두 개 이상은 절대 사용하지 마세요.
                - 질문 및 답변에서 사용된 세법 조문은 최신 데이터에 기반해야 합니다.
                - 질문이 복합적인 경우, 각 하위 질문에 대해 별도로 답변하거나, 사용자에게 우선순위를 확인합니다.

            6. 예시 답변 템플릿:
                - "질문에 대한 답변: ..."
                - "관련 세법 조항: ..."
                - "추가 설명: ..."
                - 위는 "예시" 템플릿으로, 예정 답변이 템플릿과 일치하지 않을 경우 수정 가능합니다."""
        ),
        ("human", "질문: {query}\n\n[관련 법령 및 근거 자료]\n{context}")
    ])

    if not docs:
        return "죄송합니다. 관련 근거를 찾지 못했습니다. 질문을 더 구체적으로 작성하거나 다른 키워드로 다시 시도해 주세요. 복잡한 사안이면 전문 법률 상담을 권장드립니다."

    chain = answer_prompt | llm
    response = chain.invoke({"query": query, "context": context})
    answer = response.content

    print("✅ [답변 생성 완료]")
    return answer


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
    
    # BM25 초기화
    BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "whoosh_index")
    try:
        bm25_retriever = BM25KeywordRetriever(index_dir=BM25_INDEX_DIR)
        print(f"✅ BM25/keyword 인덱스 로드 완료: {BM25_INDEX_DIR}")
    except Exception as e:
        print(f"⚠️  BM25 인덱스 로드 실패: {e}")
        bm25_retriever = None

    return {"vectorstore": vectorstore, "bm25_retriever": bm25_retriever}


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    try:
        print("\n" + "=" * 60)
        print("🚀 A-TEAM 법률 RAG 챗봇 (LangChain V3) 초기화")
        print("=" * 60 + "\n")

        resources = initialize_resources()
        vectorstore = resources["vectorstore"]
        bm25_retriever = resources["bm25_retriever"]
        
        print("\n🤖 LLM 설정 중...")
        llm = ChatOpenAI(model="gpt-4.1", temperature=0, streaming=True)
        print("✅ LLM 설정 완료")

        print("\n" + "=" * 60)
        print("✅ 🤖 A-TEAM 법률 챗봇 준비 완료 (V3)")
        print("=" * 60)
        print("\n사용 방법: 노동법/형사법/민사법 질문에 답변, 모호하면 명확화 요청")
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

                print("\n" + "-" * 60)
                print("🔄 답변 생성 중...")
                print("-" * 60 + "\n")

                # 1. 검색
                docs = retrieve_documents(user_input, vectorstore, bm25_retriever)
                
                # 2. 생성
                answer = generate_answer(user_input, docs, llm)
                
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
