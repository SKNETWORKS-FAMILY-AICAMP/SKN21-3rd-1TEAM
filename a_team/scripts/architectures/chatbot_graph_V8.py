################################################
# A-TEAM 법률 RAG 챗봇 (LangGraph V8)
# V8 리팩토링:
# - @dataclass Config로 설정 분리
# - 계층화된 구조: Infrastructure → Logic → Execution
# - 코드 가독성 향상
# 기존 기능: 질문 의도 분석, Hybrid Retriever, Query Expansion, Generator-Critic
# 작성자: SKN 3-1팀 A-TEAM
# 작성일: 2026-01-08
################################################

import os
import sys
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Annotated, TypedDict, Sequence, Optional, List, Literal, Any
)

# Third-party
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# LangChain Core
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain Retrievers
from langchain_community.retrievers import BM25Retriever

# Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# LangGraph
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END


# ============================================================
# [SECTION 1] Configuration - 모든 설정값을 한 곳에서 관리
# ============================================================
@dataclass
class Config:
    """Application Configuration (dataclass)

    모든 하드코딩된 값을 이곳에서 관리합니다.
    변경이 필요한 경우 이 클래스만 수정하면 됩니다.
    """

    # ═══════════════════════════════════════════════════════════
    # [1] Models - 사용할 모델 설정
    # ═══════════════════════════════════════════════════════════
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    RERANKER_MODEL: str = "jinaai/jina-reranker-v2-base-multilingual"

    # ═══════════════════════════════════════════════════════════
    # [2] RAG Settings - 검색 및 처리 설정
    # ═══════════════════════════════════════════════════════════
    VECTOR_DIM: int = 1024
    TOP_K_VECTOR: int = 15
    TOP_K_BM25: int = 15
    TOP_K_RERANK: int = 7
    TOP_K_FINAL: int = 5
    RELEVANCE_THRESHOLD: float = 0.2
    BM25_SAMPLE_SIZE: int = 2000
    MAX_RETRY: int = 2

    # ═══════════════════════════════════════════════════════════
    # [3] Qdrant - 벡터 DB 설정
    # ═══════════════════════════════════════════════════════════
    QDRANT_TIMEOUT: int = 30

    # ═══════════════════════════════════════════════════════════
    # [4] PROMPTS - 노드별 시스템 프롬프트
    # ═══════════════════════════════════════════════════════════

    # --- [노드: Query Expansion] 쿼리 확장용 프롬프트 ---
    PROMPT_QUERY_EXPANSION: str = """당신은 한국 법률 검색 전문가입니다.
사용자의 법률 질문을 분석하여 벡터 검색과 키워드 검색에 최적화된 쿼리로 확장합니다.

## 확장 전략
1. 핵심 키워드 추출: 질문에서 가장 중요한 법률 개념 3-5개 추출
2. 법률 용어 매핑: 일상 표현을 법률 용어로 변환
3. 관련 조항 추론: 해당 분야의 대표 법령명과 조항 추정
4. 동의어 확장: 검색 범위를 넓히기 위한 유사 표현 추가

## 출력 규칙
- expanded_query는 원본 질문 + 핵심 키워드 + 관련 법령명을 자연스럽게 조합
- 최대 100자 이내로 압축"""

    # --- [노드: Analyze] 질문 분석용 프롬프트 ---
    PROMPT_ANALYZE: str = """당신은 법률 질문을 심층 분석하는 전문가입니다.

## 분류
- category: 노동법, 형사법, 민사법, 기타
- intent_type: 법령조회, 절차문의, 상황판단, 권리확인, 분쟁해결, 일반상담
- search_strategy: 법령우선, 행정해석우선, 판례필수, 종합검색
- target_doc_types: 법, 시행령, 시행규칙, 행정해석, 판정선례

## 규칙
- needs_clarification: 1~2단어만 있어 답변 불가능한 경우에만 true
- needs_case_law: 판례 언급 또는 법적 해석 쟁점이 있는 경우 true"""

    # --- [노드: Generate] 답변 생성용 프롬프트 ---
    PROMPT_GENERATE: str = """당신은 법률 전문 AI 어시스턴트 'A-TEAM 봇'입니다.

역할:
- 검색된 법률 문서를 바탕으로 정확하고 친절하게 답변합니다.
- 법령명, 조항 등 구체적인 근거를 제시합니다.
- 법률 용어는 쉽게 풀어서 설명합니다.

답변 작성 규칙:
1. 검색된 자료를 근거로 답변하세요.
2. 답변 구조: 📌 결론 → 📖 법적 근거 → 💡 추가 설명
3. 관련 법령과 조항을 [법령명 제X조]처럼 명시하세요.
4. 확실하지 않은 내용은 "~로 해석될 수 있습니다" 등으로 신중하게 표현하세요.
5. 검색된 문서에 없는 내용은 추측하지 마세요.
6. 전문 법률 상담이 필요한 경우 안내하세요.
7. 한국어로 답변하세요."""

    # --- [노드: Evaluate] 답변 평가용 프롬프트 ---
    PROMPT_EVALUATE: str = """당신은 법률 답변의 품질을 평가하는 비평가입니다.

## 평가 기준
1. has_legal_basis: 법령명, 조항 번호 등 구체적 법적 근거 있는가
2. cites_retrieved_docs: 검색된 문서 내용이 반영되었는가
3. is_relevant: 질문에 직접 답하는가
4. needs_more_search: 검색 결과 부족하여 추가 검색 필요한가
5. quality_score: 1-5점

## 원칙
- 품질 3점 이상이면 통과, 2점 이하면 재검색 권장"""

    # --- [노드: Clarify] 명확화 요청 템플릿 ---
    TEMPLATE_CLARIFY: str = """안녕하세요! 질문을 잘 이해하기 위해 확인이 필요합니다.

{clarification_question}

위 내용을 포함해서 다시 질문해 주시면, 더 정확한 답변을 드릴 수 있습니다. 😊"""

    # --- [노드: Generate] 검색 결과 없음 시 답변 ---
    TEMPLATE_NO_RESULTS: str = """죄송합니다. 관련 법률 정보를 찾지 못했습니다.

다음과 같이 시도해 보세요:
1. 질문을 더 구체적으로 작성
2. 다른 키워드로 질문
3. 전문 법률 상담 권장

📌 참고: https://law.go.kr"""


# ============================================================
# [SECTION 2] Logging Setup
# ============================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LegalRAG-V8")


# ============================================================
# [SECTION 3] State Definition - LangGraph 상태 정의
# ============================================================
class AgentState(TypedDict):
    """LangGraph Agent의 상태"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    query_analysis: Optional[dict]
    retrieved_docs: Optional[List[Document]]
    generated_answer: Optional[str]
    next_action: Optional[str]
    evaluation_result: Optional[dict]
    retry_count: Optional[int]


# ============================================================
# [SECTION 4] Reranker - 커스텀 Jina Reranker Wrapper
# ============================================================
class JinaReranker(BaseDocumentCompressor):
    """Jina Reranker Wrapper for LangChain"""
    model_name: str = "jinaai/jina-reranker-v2-base-multilingual"
    top_n: int = 7
    model: Any = None
    tokenizer: Any = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, model_name: str = None, top_n: int = None, **kwargs):
        super().__init__(**kwargs)
        if model_name:
            self.model_name = model_name
        if top_n:
            self.top_n = top_n

        logger.info(f"Loading Reranker: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype="auto"
        )
        self.model.eval()
        logger.info("Reranker loaded successfully")

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Optional[Any] = None
    ) -> Sequence[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            )
            scores = self.model(**inputs).logits.squeeze(-1).float().cpu()
            scores = torch.sigmoid(scores).tolist()
            if not isinstance(scores, list):
                scores = [scores]

        # Sort and select top_n
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:self.top_n]

        final_docs = []
        for i in top_indices:
            doc = documents[i]
            doc.metadata["relevance_score"] = scores[i]
            final_docs.append(doc)

        return final_docs


# ============================================================
# [SECTION 5] Pydantic Schemas - LLM 구조화된 출력용
# ============================================================
class ExpandedQuery(BaseModel):
    """검색 쿼리 확장 결과"""
    original_query: str = Field(description="원본 사용자 질문")
    search_keywords: List[str] = Field(description="핵심 검색 키워드 (3-5개)")
    legal_terms: List[str] = Field(description="관련 법률 용어 및 조항명")
    synonyms: List[str] = Field(description="동의어 및 유사 표현 (2-3개)")
    expanded_query: str = Field(description="확장된 검색 쿼리")


class QueryAnalysis(BaseModel):
    """질문 분석 결과"""
    category: str = Field(description="법률 분야: 노동법, 형사법, 민사법, 기타")
    needs_clarification: bool = Field(default=False, description="질문 모호 여부")
    needs_case_law: bool = Field(default=False, description="판례 검색 필요 여부")
    clarification_question: str = Field(default="", description="명확화 질문")
    intent_type: str = Field(
        description="질문 의도: 법령조회, 절차문의, 상황판단, 권리확인, 분쟁해결, 일반상담")
    user_situation: str = Field(default="", description="사용자 상황 요약")
    core_question: str = Field(default="", description="핵심 질문")
    search_strategy: str = Field(description="검색 전략: 법령우선, 행정해석우선, 판례필수, 종합검색")
    target_doc_types: List[str] = Field(
        default_factory=list, description="검색 대상 문서 타입")
    related_laws: List[str] = Field(default_factory=list, description="관련 법률명")


class AnswerEvaluation(BaseModel):
    """답변 평가 결과"""
    has_legal_basis: bool = Field(description="법적 근거 명시 여부")
    cites_retrieved_docs: bool = Field(description="검색 문서 인용 여부")
    is_relevant: bool = Field(description="답변 적합성")
    needs_more_search: bool = Field(description="추가 검색 필요 여부")
    quality_score: int = Field(description="품질 점수 (1-5)")
    improvement_suggestion: str = Field(default="", description="개선 제안")


# ============================================================
# [SECTION 6] Infrastructure Layer - 외부 리소스 연결
# ============================================================
class VectorStoreManager:
    """Qdrant 벡터스토어 관리"""

    def __init__(self, config: Config):
        self.config = config
        self._load_env()
        self.embeddings = None
        self.vectorstore = None
        self.client = None

    def _load_env(self):
        """환경 변수 로드"""
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY가 .env에 설정되지 않았습니다!")

    def initialize(self) -> QdrantVectorStore:
        """벡터스토어 초기화"""
        logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding model loaded")

        logger.info("Connecting to Qdrant...")
        warnings.filterwarnings(
            'ignore', message='Api key is used with an insecure connection')

        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=self.config.QDRANT_TIMEOUT,
            prefer_grpc=False
        )
        logger.info("Qdrant connected")

        logger.info("Initializing vectorstore...")
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            content_payload_key="text"
        )
        logger.info("Vectorstore initialized")

        return self.vectorstore

    def get_client(self) -> QdrantClient:
        return self.client

    def get_collection_name(self) -> str:
        return self.collection_name


class BM25Manager:
    """BM25 Retriever 관리"""

    def __init__(self, config: Config, client: QdrantClient, collection_name: str):
        self.config = config
        self.client = client
        self.collection_name = collection_name
        self.retriever = None

    def initialize(self) -> Optional[BM25Retriever]:
        """BM25 Retriever 초기화"""
        logger.info("Initializing BM25 Retriever...")

        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            logger.info(f"Collection contains {total_points} documents")

            sample_size = min(self.config.BM25_SAMPLE_SIZE, total_points)
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=sample_size,
                with_payload=True,
                with_vectors=False
            )

            bm25_docs = []
            for point in scroll_result[0]:
                payload = point.payload
                text = payload.get("text", "")
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={k: v for k, v in payload.items()
                                  if k != "text"}
                    )
                    bm25_docs.append(doc)

            if bm25_docs:
                self.retriever = BM25Retriever.from_documents(
                    bm25_docs, k=self.config.TOP_K_BM25)
                logger.info(
                    f"BM25 Retriever initialized ({len(bm25_docs)} docs)")
                return self.retriever
            else:
                logger.warning("No documents for BM25. Vector Search only.")
                return None

        except Exception as e:
            logger.error(f"BM25 init failed: {e}")
            return None


# ============================================================
# [SECTION 7] Logic Layer - LangGraph 노드 및 워크플로우 구성
# ============================================================
class LegalRAGBuilder:
    """법률 RAG 그래프 빌더"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.vectorstore = None
        self.bm25_retriever = None
        self.query_expander = None
        self.reranker = None

    def _init_infrastructure(self):
        """인프라 초기화"""
        # Vector Store
        vs_manager = VectorStoreManager(self.config)
        self.vectorstore = vs_manager.initialize()

        # BM25
        bm25_manager = BM25Manager(
            self.config,
            vs_manager.get_client(),
            vs_manager.get_collection_name()
        )
        self.bm25_retriever = bm25_manager.initialize()

        # LLM
        logger.info(f"Initializing LLM: {self.config.LLM_MODEL}")
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE,
            streaming=True
        )

        # Query Expander
        self.query_expander = self._create_query_expander()

        # Reranker
        self.reranker = JinaReranker(
            model_name=self.config.RERANKER_MODEL,
            top_n=self.config.TOP_K_RERANK
        )

    def _create_query_expander(self):
        """Query Expander 생성 [사용 프롬프트: PROMPT_QUERY_EXPANSION]"""
        structured_llm = self.llm.with_structured_output(ExpandedQuery)

        expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.PROMPT_QUERY_EXPANSION),
            ("human", "{query}")
        ])

        def expand_query(query: str) -> ExpandedQuery:
            try:
                chain = expansion_prompt | structured_llm
                return chain.invoke({"query": query})
            except Exception:
                return ExpandedQuery(
                    original_query=query, search_keywords=[],
                    legal_terms=[], synonyms=[], expanded_query=query
                )

        return expand_query

    # --- Nodes ---

    def _create_analyze_node(self):
        """[노드: Analyze] 질문 분석 노드 [사용 프롬프트: PROMPT_ANALYZE]"""
        structured_llm = self.llm.with_structured_output(QueryAnalysis)

        analyze_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.PROMPT_ANALYZE),
            ("human", "{query}")
        ])

        def analyze_query(state: AgentState) -> AgentState:
            query = state["user_query"]
            logger.info(f"Analyzing query: {query[:50]}...")

            chain = analyze_prompt | structured_llm
            analysis: QueryAnalysis = chain.invoke({"query": query})

            logger.info(
                f"Analysis: category={analysis.category}, intent={analysis.intent_type}")

            return {"query_analysis": analysis.model_dump()}

        return analyze_query

    def _create_clarify_node(self):
        """[노드: Clarify] 명확화 요청 노드 [사용 템플릿: TEMPLATE_CLARIFY]"""
        template = self.config.TEMPLATE_CLARIFY

        def request_clarification(state: AgentState) -> AgentState:
            analysis = state.get("query_analysis", {})
            clarification_q = analysis.get(
                "clarification_question", "질문을 좀 더 구체적으로 해주시겠어요?")

            answer = template.format(clarification_question=clarification_q)
            return {"generated_answer": answer, "next_action": "end"}

        return request_clarification

    def _create_search_node(self):
        """하이브리드 검색 노드"""
        vectorstore = self.vectorstore
        bm25_retriever = self.bm25_retriever
        query_expander = self.query_expander
        reranker = self.reranker
        config = self.config

        def search_documents(state: AgentState) -> AgentState:
            original_query = state["user_query"]
            analysis = state.get("query_analysis", {})
            related_laws = analysis.get("related_laws", [])

            # Query Expansion
            if query_expander:
                try:
                    expanded = query_expander(original_query)
                    search_query = expanded.expanded_query
                    logger.info(f"Expanded query: {search_query[:60]}...")
                except Exception:
                    search_query = original_query
            else:
                search_query = original_query

            all_docs = []

            # 1. Vector Search
            try:
                vector_results = vectorstore.similarity_search_with_score(
                    search_query, k=config.TOP_K_VECTOR)
                vector_docs = [doc for doc, _ in vector_results]
                for doc in vector_docs:
                    doc.metadata["search_source"] = "vector"
                all_docs.extend(vector_docs)
                logger.info(f"Vector search: {len(vector_docs)} docs")
            except Exception as e:
                logger.error(f"Vector search error: {e}")

            # 2. BM25 Search
            if bm25_retriever:
                try:
                    bm25_docs = bm25_retriever.invoke(search_query)
                    for doc in bm25_docs:
                        doc.metadata["search_source"] = "bm25"
                    all_docs.extend(bm25_docs)
                    logger.info(f"BM25 search: {len(bm25_docs)} docs")
                except Exception as e:
                    logger.error(f"BM25 search error: {e}")

            # 3. Deduplicate
            seen = set()
            unique_docs = []
            for doc in all_docs:
                h = hash(doc.page_content[:200])
                if h not in seen:
                    seen.add(h)
                    unique_docs.append(doc)

            logger.info(f"After dedup: {len(unique_docs)} docs")

            if not unique_docs:
                return {"retrieved_docs": []}

            # 4. Rerank
            try:
                reranked_docs = reranker.compress_documents(
                    unique_docs, original_query)

                # 5. Boost related laws
                if related_laws:
                    for doc in reranked_docs:
                        law_name = doc.metadata.get('law_name', '')
                        for rel_law in related_laws:
                            if rel_law in law_name:
                                score = doc.metadata.get('relevance_score', 0)
                                doc.metadata['relevance_score'] = min(
                                    1.0, score + 0.1)
                                doc.metadata['boosted'] = True
                                break

                # 6. Filter by threshold
                filtered_docs = [
                    doc for doc in reranked_docs
                    if doc.metadata.get('relevance_score', 0) >= config.RELEVANCE_THRESHOLD
                ]

                logger.info(f"After rerank/filter: {len(filtered_docs)} docs")

                return {"retrieved_docs": filtered_docs[:config.TOP_K_FINAL]}

            except Exception as e:
                logger.error(f"Rerank error: {e}")
                return {"retrieved_docs": unique_docs[:config.TOP_K_FINAL]}

        return search_documents

    def _create_generate_node(self):
        """[노드: Generate] 답변 생성 노드 [사용 프롬프트: PROMPT_GENERATE, TEMPLATE_NO_RESULTS]"""
        llm = self.llm
        system_prompt = self.config.PROMPT_GENERATE
        no_results_template = self.config.TEMPLATE_NO_RESULTS

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """사용자 질문: {query}

📚 검색된 법령/문서:
{context}

{case_law_notice}

위 자료를 바탕으로 질문에 답변해주세요.""")
        ])

        def generate_answer(state: AgentState) -> AgentState:
            query = state["user_query"]
            docs = state.get("retrieved_docs", [])
            analysis = state.get("query_analysis", {})
            needs_case_law = analysis.get("needs_case_law", False)

            logger.info("Generating answer...")

            # Format context
            if docs:
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    meta = doc.metadata
                    law_name = meta.get("law_name", "")
                    article = meta.get("article_no", "")
                    title = meta.get(
                        "article_title", "") or meta.get("title", "")
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

            case_law_notice = ""
            if needs_case_law:
                case_law_notice = "⚠️ 참고: 판례 검색이 필요하나 현재 DB에 포함되어 있지 않습니다."

            if not docs:
                answer = no_results_template
            else:
                chain = answer_prompt | llm
                response = chain.invoke({
                    "query": query,
                    "context": context,
                    "case_law_notice": case_law_notice
                })
                answer = response.content

            logger.info("Answer generated")
            return {"generated_answer": answer}

        return generate_answer

    def _create_evaluate_node(self):
        """[노드: Evaluate] 답변 평가 노드 [사용 프롬프트: PROMPT_EVALUATE]"""
        structured_llm = self.llm.with_structured_output(AnswerEvaluation)

        evaluate_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.PROMPT_EVALUATE),
            ("human", """## 질문
{query}

## 검색된 문서 요약
{context_summary}

## 생성된 답변
{answer}

평가해주세요.""")
        ])

        def evaluate_answer(state: AgentState) -> AgentState:
            query = state["user_query"]
            answer = state.get("generated_answer", "")
            docs = state.get("retrieved_docs", [])
            retry_count = state.get("retry_count", 0) or 0

            logger.info(f"Evaluating answer (attempt {retry_count + 1})")

            if docs:
                context_summary = "\n".join([
                    f"- {doc.metadata.get('law_name', '문서')}: {doc.page_content[:100]}..."
                    for doc in docs[:5]
                ])
            else:
                context_summary = "(검색된 문서 없음)"

            chain = evaluate_prompt | structured_llm
            evaluation: AnswerEvaluation = chain.invoke({
                "query": query,
                "context_summary": context_summary,
                "answer": answer
            })

            logger.info(
                f"Evaluation: score={evaluation.quality_score}, needs_more={evaluation.needs_more_search}")

            return {
                "evaluation_result": evaluation.model_dump(),
                "retry_count": retry_count + 1
            }

        return evaluate_answer

    # --- Routing ---

    def _route_after_analysis(self, state: AgentState) -> Literal["clarify", "search"]:
        analysis = state.get("query_analysis", {})
        if analysis.get("needs_clarification", False):
            return "clarify"
        return "search"

    def _route_after_evaluation(self, state: AgentState) -> Literal["search", "end"]:
        evaluation = state.get("evaluation_result", {})
        retry_count = state.get("retry_count", 0) or 0

        if retry_count >= self.config.MAX_RETRY:
            logger.warning("Max retry reached")
            return "end"

        if evaluation.get("needs_more_search", False) and evaluation.get("quality_score", 3) <= 2:
            logger.info("Retrying search...")
            return "search"

        return "end"

    # --- Build Graph ---

    def build(self) -> StateGraph:
        """그래프 빌드"""
        logger.info("Building Legal RAG Graph...")

        # Infrastructure
        self._init_infrastructure()

        # Create nodes
        analyze_node = self._create_analyze_node()
        clarify_node = self._create_clarify_node()
        search_node = self._create_search_node()
        generate_node = self._create_generate_node()
        evaluate_node = self._create_evaluate_node()

        # Build workflow
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze", analyze_node)
        workflow.add_node("clarify", clarify_node)
        workflow.add_node("search", search_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("evaluate", evaluate_node)

        workflow.set_entry_point("analyze")

        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analysis,
            {"clarify": "clarify", "search": "search"}
        )

        workflow.add_edge("clarify", END)
        workflow.add_edge("search", "generate")
        workflow.add_edge("generate", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {"search": "search", "end": END}
        )

        graph = workflow.compile()
        logger.info("Graph built successfully")

        return graph


# ============================================================
# [SECTION 8] Execution Layer - 진입점 및 실행 로직
# ============================================================

# 환경 변수 로드
_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH)

# 전역 Config
config = Config()


def initialize_rag_chatbot():
    """평가 스크립트용 초기화 함수"""
    builder = LegalRAGBuilder(config)
    return builder.build()


def main():
    """메인 실행 함수"""
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set")
        return

    print("\n" + "=" * 60)
    print("🚀 A-TEAM 법률 RAG 챗봇 (LangGraph V8) 시작")
    print("=" * 60 + "\n")

    try:
        graph = initialize_rag_chatbot()

        print("\n✅ 챗봇 준비 완료!")
        print("💡 'exit' 또는 '종료'를 입력하면 종료됩니다\n")

        while True:
            try:
                user_input = input("👤 User >> ").strip()

                if user_input.lower() in ["exit", "quit", "종료", "q"]:
                    print("\n👋 챗봇을 종료합니다.")
                    break

                if not user_input:
                    print("❌ 질문을 입력해주세요.\n")
                    continue

                initial_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "user_query": user_input,
                    "query_analysis": None,
                    "retrieved_docs": None,
                    "generated_answer": None,
                    "next_action": None,
                    "evaluation_result": None,
                    "retry_count": 0
                }

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
                print("\n\n👋 챗봇을 종료합니다.")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ 오류 발생: {e}\n")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise


if __name__ == "__main__":
    main()
