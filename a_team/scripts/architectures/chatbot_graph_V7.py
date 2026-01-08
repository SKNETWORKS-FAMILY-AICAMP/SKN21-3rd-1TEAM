################################################
# A-TEAM 법률 RAG 챗봇 (LangGraph V7)
# V7 신규 기능:
# - 질문 의도 분석 (6개 유형: 법령조회, 절차문의, 상황판단, 권리확인, 분쟁해결, 일반상담)
# - 검색 전략 결정 (법령우선, 행정해석우선, 판례필수, 종합검색)
# 기존 기능: Hybrid Retriever, Query Expansion, Generator-Critic Light
# 작성자: SKN 3-1팀 A-TEAM
# 작성일: 2026-01-08
################################################

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from typing import Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pydantic import BaseModel, Field
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Sequence, Optional, List, Literal
from pathlib import Path
import warnings
import os

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
    # 생성된 답변
    generated_answer: Optional[str]
    # 현재 라우팅 결정
    next_action: Optional[str]
    # [V5] 답변 평가 결과 (Generator-Critic)
    evaluation_result: Optional[dict]
    # [V5] 재검색 시도 횟수 (무한 루프 방지)
    retry_count: Optional[int]


# ===========================
# Reranker 정의
# ===========================
class JinaReranker(BaseDocumentCompressor):
    model_name: str = "jinaai/jina-reranker-v2-base-multilingual"
    top_n: int = 7
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

# Pydantic 모델: Query Expansion 결과
class ExpandedQuery(BaseModel):
    """검색 쿼리 확장 결과"""
    original_query: str = Field(description="원본 사용자 질문")
    search_keywords: List[str] = Field(description="핵심 검색 키워드 (3-5개)")
    legal_terms: List[str] = Field(
        description="관련 법률 용어 및 조항명 (예: 근로기준법 제23조)")
    synonyms: List[str] = Field(description="동의어 및 유사 표현 (2-3개)")
    expanded_query: str = Field(description="확장된 검색 쿼리 (원본 + 키워드 조합)")


def create_query_expander(llm: ChatOpenAI):
    """Query Expansion 함수 생성 - 법률 도메인 특화"""

    structured_llm = llm.with_structured_output(ExpandedQuery)

    expansion_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 한국 법률 검색 전문가입니다. 
사용자의 법률 질문을 분석하여 벡터 검색과 키워드 검색에 최적화된 쿼리로 확장합니다.

## 목표
법률 데이터베이스에서 관련 문서를 최대한 많이 검색할 수 있도록 쿼리를 확장합니다.

## 확장 전략
1. **핵심 키워드 추출**: 질문에서 가장 중요한 법률 개념 3-5개 추출
2. **법률 용어 매핑**: 일상 표현을 법률 용어로 변환 (예: "월급" → "임금", "잘림" → "해고")
3. **관련 조항 추론**: 해당 분야의 대표 법령명과 조항 추정 (예: "주휴수당" → "근로기준법 제55조")
4. **동의어 확장**: 검색 범위를 넓히기 위한 유사 표현 추가

## 법률 분야별 주요 키워드
- 노동법: 근로기준법, 임금, 퇴직금, 해고, 산재, 주휴수당, 연차, 근로계약
- 형사법: 형법, 형사소송법, 고소, 고발, 기소, 구속, 공소시효
- 민사법: 민법, 계약, 손해배상, 소유권, 채권, 물권, 불법행위

## 출력 규칙
- expanded_query는 원본 질문 + 핵심 키워드 + 관련 법령명을 자연스럽게 조합
- 검색에 불필요한 조사, 어미는 제거
- 최대 100자 이내로 압축"""),
        ("human", "{query}")
    ])

    def expand_query(query: str) -> ExpandedQuery:
        """질문을 검색에 최적화된 형태로 확장"""
        try:
            chain = expansion_prompt | structured_llm
            result: ExpandedQuery = chain.invoke({"query": query})
            return result
        except Exception as e:
            # 실패 시 원본 쿼리 그대로 반환
            return ExpandedQuery(
                original_query=query,
                search_keywords=[],
                legal_terms=[],
                synonyms=[],
                expanded_query=query
            )

    return expand_query


# Pydantic 모델: 질문 분석 결과 (V7 - 의도 분석 + 검색 전략)
class QueryAnalysis(BaseModel):
    """LLM이 반환할 질문 분석 결과 - V7 확장"""
    # 기본 분류
    category: str = Field(description="법률 분야: 노동법, 형사법, 민사법, 기타 중 하나")
    needs_clarification: bool = Field(
        default=False, description="질문이 극도로 모호하여 답변 불가능한지")
    needs_case_law: bool = Field(default=False, description="대법원 판례 검색이 필요한지")
    clarification_question: str = Field(
        default="", description="명확화 필요 시 사용자에게 물어볼 질문")

    # [V7] 질문 의도 분석
    intent_type: str = Field(
        description="질문 의도: 법령조회, 절차문의, 상황판단, 권리확인, 분쟁해결, 일반상담 중 하나")
    user_situation: str = Field(
        default="", description="사용자가 처한 상황 1-2문장 요약")
    core_question: str = Field(
        default="", description="질문의 핵심 (한 문장으로 추출)")

    # [V7] 검색 전략
    search_strategy: str = Field(
        description="검색 전략: 법령우선, 행정해석우선, 판례필수, 종합검색 중 하나")
    target_doc_types: List[str] = Field(
        default_factory=list,
        description="검색할 문서 타입 리스트: 법, 시행령, 시행규칙, 행정해석, 판정선례 중 선택")
    related_laws: List[str] = Field(
        default_factory=list,
        description="예상 관련 법률명 (예: 근로기준법, 산업재해보상보험법)")


def create_analyze_query_node(llm: ChatOpenAI):
    """노드 1: 질문 분석 (V7 - 의도 분석 + 검색 전략)"""

    # Structured Output을 위한 LLM
    structured_llm = llm.with_structured_output(QueryAnalysis)

    analyze_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 질문을 심층 분석하는 전문가입니다. 사용자의 질문을 분석하여 의도를 파악하고 최적의 검색 전략을 결정합니다.

## 1. 기본 분류
**category** (법률 분야):
- "노동법": 근로기준법, 임금, 퇴직금, 해고, 산재, 주휴수당, 연차휴가, 근로계약 등
- "형사법": 범죄, 형벌, 수사, 재판, 고소/고발, 형사소송 등
- "민사법": 계약, 손해배상, 소유권, 채권, 불법행위, 민사소송 등
- "기타": 위 카테고리에 속하지 않는 법률 질문

**needs_clarification**: 
- true: 1~2단어만 있어 어떤 답변도 불가능한 경우 ("법률", "계약", "도와줘")
- false: 상황이 조금이라도 설명되어 있으면 답변 가능

**needs_case_law**: 
- true: "판례", "판결", "대법원" 언급 또는 법적 해석이 필요한 쟁점
- false: 단순 법령 조회, 절차 문의

## 2. 질문 의도 분석 (intent_type)
사용자가 무엇을 원하는지 파악:
- **"법령조회"**: 특정 법령, 조항, 규정의 내용을 알고 싶음 (예: "근로기준법 제23조가 뭐야?")
- **"절차문의"**: 신청, 접수, 처리 절차를 알고 싶음 (예: "산재 신청 어떻게 해?")
- **"상황판단"**: 자신의 상황이 법적으로 어떤 상태인지 판단 요청 (예: "이게 부당해고야?")
- **"권리확인"**: 자신에게 어떤 권리가 있는지 확인 (예: "퇴직금 받을 수 있어?")
- **"분쟁해결"**: 갈등/분쟁 상황에서 해결 방법 문의 (예: "사장이 임금 안 줘 어떻게 해?")
- **"일반상담"**: 위에 해당하지 않는 일반적 법률 질문

## 3. 상황 분석
**user_situation**: 사용자가 처한 상황을 1-2문장으로 요약
**core_question**: 질문의 핵심을 한 문장으로 추출

## 4. 검색 전략 결정 (search_strategy)
질문 유형에 따라 최적의 검색 전략 선택:
- **"법령우선"**: 법령조회, 권리확인 → 법 조문이 가장 중요
- **"행정해석우선"**: 절차문의 → 행정해석/시행규칙이 실무적
- **"판례필수"**: 상황판단에서 쟁점이 있거나 needs_case_law가 true
- **"종합검색"**: 분쟁해결, 복합적 질문 → 다양한 문서 필요

## 5. 문서 타입 추천 (target_doc_types)
검색할 문서 타입 선택 (복수 선택 가능):
- "법": 기본 법률 (근로기준법, 형법 등)
- "시행령": 대통령령 (법의 세부 시행사항)
- "시행규칙": 부령 (절차, 서식, 기준)
- "행정해석": 고용노동부 등 행정기관 해석
- "판정선례": 노동위원회 등 판정 사례

## 6. 관련 법률 추론 (related_laws)
질문에서 예상되는 관련 법률명 나열 (예: ["근로기준법", "산업재해보상보험법"])"""),
        ("human", "{query}")
    ])

    def analyze_query(state: AgentState) -> AgentState:
        """질문 분석 노드: 의도 파악 + 검색 전략 결정"""
        query = state["user_query"]

        print(f"🔎 [질문 심층 분석 중...]")

        chain = analyze_prompt | structured_llm
        analysis: QueryAnalysis = chain.invoke({"query": query})

        # 분석 결과 상세 출력
        print(f"")
        print(f"📋 [분석 결과]")
        print(f"   📂 분야: {analysis.category}")
        print(f"   🎯 의도: {analysis.intent_type}")
        print(
            f"   💭 상황: {analysis.user_situation[:50]}..." if analysis.user_situation else "   💭 상황: (없음)")
        print(f"   ❓ 핵심 질문: {analysis.core_question}")
        print(f"   🔍 검색 전략: {analysis.search_strategy}")
        print(f"   📑 대상 문서: {', '.join(analysis.target_doc_types)}")
        print(f"   📚 관련 법률: {', '.join(analysis.related_laws)}")
        print(f"   명확화 필요: {'예' if analysis.needs_clarification else '아니오'}")
        print(f"   판례 필요: {'예' if analysis.needs_case_law else '아니오'}")
        print(f"")

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


def create_search_node(vectorstore: QdrantVectorStore,
                       bm25_retriever: Optional[BM25Retriever] = None,
                       query_expander=None):
    """노드 3: 하이브리드 검색 (V7 - 의도 기반 필터링 + 법률 부스팅)"""

    # Reranker를 한 번만 생성 (성능 최적화)
    _reranker = JinaReranker(top_n=7)  # V7: top_n을 7로 증가 (필터링 후 5개 유지)

    def get_doc_type(law_name: str) -> str:
        """법령명에서 문서 타입 추론"""
        if '시행규칙' in law_name:
            return '시행규칙'
        elif '시행령' in law_name:
            return '시행령'
        elif law_name:  # 기본 법률
            return '법'
        return '기타'

    def search_documents(state: AgentState) -> AgentState:
        """검색 실행 노드: V7 - 의도 기반 필터링 + 법률 부스팅"""
        original_query = state["user_query"]

        # [V7] 분석 결과에서 검색 전략 추출
        analysis = state.get("query_analysis", {})
        search_strategy = analysis.get("search_strategy", "종합검색")
        target_doc_types = analysis.get("target_doc_types", [])
        related_laws = analysis.get("related_laws", [])

        print(f"🎯 [V7 검색 전략] {search_strategy}")
        if target_doc_types:
            print(f"   📑 대상 문서: {', '.join(target_doc_types)}")
        if related_laws:
            print(f"   📚 관련 법률: {', '.join(related_laws)}")

        # Query Expansion 적용
        if query_expander is not None:
            print(f"🔍 [Query Expansion] 쿼리 확장 중...")
            try:
                expanded = query_expander(original_query)
                search_query = expanded.expanded_query
                print(f"   📝 원본: {original_query[:40]}...")
                print(f"   🔄 확장: {search_query[:60]}...")
                if expanded.legal_terms:
                    print(f"   📋 법률 용어: {', '.join(expanded.legal_terms[:3])}")
            except Exception as e:
                print(f"   ⚠️  Query Expansion 실패: {e}")
                search_query = original_query
        else:
            search_query = original_query

        print(f"🔍 [하이브리드 검색] 쿼리: {search_query[:50]}...")

        all_docs = []

        # 1. Vector Search (유사도 기반) - 확장된 쿼리 사용
        print(f"   📊 [Vector Search] 실행 중...")
        try:
            vector_results = vectorstore.similarity_search_with_score(
                search_query, k=15)
            vector_docs = [doc for doc, score in vector_results]
            print(f"   ✅ Vector Search: {len(vector_docs)}개 문서 검색")

            # Vector 검색 결과에 source 표시
            for doc in vector_docs:
                doc.metadata["search_source"] = "vector"
            all_docs.extend(vector_docs)
        except Exception as e:
            print(f"   ⚠️  Vector Search 오류: {e}")

        # 2. BM25 Search (키워드 기반) - 확장된 쿼리 사용
        if bm25_retriever is not None:
            print(f"   📝 [BM25 Search] 실행 중...")
            try:
                bm25_docs = bm25_retriever.invoke(search_query)
                print(f"   ✅ BM25 Search: {len(bm25_docs)}개 문서 검색")

                # BM25 검색 결과에 source 표시
                for doc in bm25_docs:
                    doc.metadata["search_source"] = "bm25"
                all_docs.extend(bm25_docs)
            except Exception as e:
                print(f"   ⚠️  BM25 Search 오류: {e}")
        else:
            print(f"   ⚠️  BM25 Retriever 미설정 (Vector Search만 사용)")

        # 3. 중복 제거 (page_content 기준)
        seen_contents = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:200])  # 앞 200자로 중복 체크
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)

        print(f"   🔄 중복 제거 후: {len(unique_docs)}개 문서")

        # 유사도 임계값 설정 (이 점수 미만은 관련 없는 문서로 판단)
        RELEVANCE_THRESHOLD = 0.2

        if unique_docs:
            # 4. 리랭킹 (Jina Reranker) - 원본 쿼리로 리랭킹 (의미 보존)
            print(f"🔄 [리랭킹] Jina Reranker로 상위 문서 선별 중...")
            try:
                reranked_docs = _reranker.compress_documents(
                    unique_docs, original_query)

                if reranked_docs:
                    # [V7] 5. 관련 법률 부스팅 - related_laws에 해당하는 문서 점수 상향
                    if related_laws:
                        print(f"   🚀 [법률 부스팅] 관련 법률 문서 점수 상향...")
                        for doc in reranked_docs:
                            law_name = doc.metadata.get('law_name', '')
                            for rel_law in related_laws:
                                if rel_law in law_name:
                                    original_score = doc.metadata.get(
                                        'relevance_score', 0)
                                    boosted_score = min(
                                        1.0, original_score + 0.1)
                                    doc.metadata['relevance_score'] = boosted_score
                                    doc.metadata['boosted'] = True
                                    print(
                                        f"      ↑ {law_name}: {original_score:.3f} → {boosted_score:.3f}")
                                    break

                    # 6. 유사도 임계값 필터링 - 낮은 점수 문서 제외
                    filtered_docs = []
                    for doc in reranked_docs:
                        score = doc.metadata.get('relevance_score', 0)
                        if score >= RELEVANCE_THRESHOLD:
                            filtered_docs.append(doc)

                    # [V7] 7. 문서 타입 필터링 제거됨 - 리랭커 점수 순서 유지

                    print(
                        f"✅ [리랭킹 완료] {len(reranked_docs)}개 → {len(filtered_docs)}개 (임계값 {RELEVANCE_THRESHOLD} 이상)")
                    for i, doc in enumerate(filtered_docs[:5], 1):
                        source = doc.metadata.get('search_source', 'unknown')
                        doc_type = doc.metadata.get('doc_type', '')
                        boosted = "⬆️" if doc.metadata.get('boosted') else ""
                        print(
                            f"   [{i}] 점수: {doc.metadata.get('relevance_score', 0):.4f} {boosted} | {doc_type} | {doc.page_content[:30]}...")

                    if filtered_docs:
                        docs = filtered_docs[:5]  # 최종 상위 5개
                    else:
                        # 임계값 통과 문서가 없으면 빈 리스트 (관련 문서 없음)
                        print(
                            f"⚠️  [관련 문서 없음] 모든 문서의 유사도가 {RELEVANCE_THRESHOLD} 미만입니다")
                        docs = []
                else:
                    print(f"⚠️  [리랭킹 결과 없음] 원본 검색 결과 사용 (상위 5개)")
                    docs = unique_docs[:5]
            except Exception as e:
                print(f"⚠️  [리랭킹 오류] {e}")
                print(f"   원본 검색 결과 사용 (상위 5개)")
                docs = unique_docs[:5]

            if docs:
                print(f"✅ [하이브리드 검색 완료] {len(docs)}개 관련 문서")
            else:
                print(f"⚠️  [검색 완료] 관련 문서 없음")
        else:
            docs = []
            print(f"⚠️  [검색 결과 없음]")

        return {
            "retrieved_docs": docs
        }

    return search_documents


def create_generate_node(llm: ChatOpenAI):
    """노드 4: 최종 답변 생성 (DB 검색 결과만 사용)"""

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 전문 AI 어시스턴트 'A-TEAM 봇'입니다.

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
7. 한국어로 답변하세요."""),
        ("human", """사용자 질문: {query}

📚 검색된 법령/문서:
{context}

{case_law_notice}

위 자료를 바탕으로 질문에 답변해주세요.""")
    ])

    def generate_answer(state: AgentState) -> AgentState:
        """답변 생성 노드: 검색 결과를 종합하여 답변 생성"""
        query = state["user_query"]
        docs = state.get("retrieved_docs", [])
        analysis = state.get("query_analysis", {})
        needs_case_law = analysis.get("needs_case_law", False)

        print(f"💬 [답변 생성 중...]")

        # 문서 컨텍스트 포맷팅
        if docs:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
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

        # 판례 필요 여부 안내 (DB에 판례가 없으므로 안내)
        if needs_case_law:
            case_law_notice = "⚠️ 참고: 사용자가 판례 정보를 요청했으나, 현재 데이터베이스에 해당 판례 정보가 포함되어 있지 않습니다. 대법원 종합법률정보(https://glaw.scourt.go.kr)에서 직접 검색하시기 바랍니다."
        else:
            case_law_notice = ""

        # 검색 결과가 전혀 없는 경우
        if not docs:
            answer = """죄송합니다. 질문과 관련된 법률 정보를 데이터베이스에서 찾지 못했습니다.

다음과 같이 시도해 보시겠어요?
1. 질문을 더 구체적으로 작성해 주세요 (예: 상황, 관련 법령 등)
2. 다른 키워드로 질문해 보세요
3. 복잡한 사안의 경우 전문 법률 상담을 권장드립니다.

📌 참고 사이트:
- 법제처 국가법령정보센터: https://law.go.kr
- 대법원 종합법률정보: https://glaw.scourt.go.kr"""
        else:
            # LLM으로 답변 생성
            chain = answer_prompt | llm
            response = chain.invoke({
                "query": query,
                "context": context,
                "case_law_notice": case_law_notice
            })
            answer = response.content

        print(f"✅ [답변 생성 완료]")

        return {
            "generated_answer": answer
        }

    return generate_answer


# Pydantic 모델: 답변 평가 결과 (Generator-Critic Light)
class AnswerEvaluation(BaseModel):
    """LLM이 반환할 답변 평가 결과"""
    has_legal_basis: bool = Field(description="답변에 법적 근거(법령, 조항)가 명시되어 있는가")
    cites_retrieved_docs: bool = Field(description="검색된 문서 내용을 실제로 인용했는가")
    is_relevant: bool = Field(description="답변이 질문에 적절히 대응하는가")
    needs_more_search: bool = Field(description="더 나은 답변을 위해 추가 검색이 필요한가")
    quality_score: int = Field(description="답변 품질 점수 (1-5, 5가 최고)")
    improvement_suggestion: str = Field(
        default="", description="개선이 필요한 경우 구체적 제안")


def create_evaluate_node(llm: ChatOpenAI):
    """노드 6: 답변 품질 평가 (Generator-Critic Light)"""

    structured_llm = llm.with_structured_output(AnswerEvaluation)

    evaluate_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 법률 답변의 품질을 평가하는 비평가(Critic)입니다.

## 평가 기준
1. **has_legal_basis**: 답변에 법령명, 조항 번호, 판례 번호 등 구체적 법적 근거가 있는가
2. **cites_retrieved_docs**: 검색된 문서의 내용이 답변에 실제로 반영되었는가
3. **is_relevant**: 사용자의 질문에 직접적으로 답하고 있는가
4. **needs_more_search**: 검색 결과가 부족하여 추가 검색이 필요한가
   - true: 검색된 문서가 질문과 관련 없거나, 답변에 "찾지 못했습니다" 등이 포함된 경우
   - false: 충분한 근거가 있거나, 이미 최선의 답변인 경우
5. **quality_score**: 1-5점 (1: 매우 부족, 3: 보통, 5: 매우 우수)

## 판단 원칙
- 법률 답변은 정확성이 생명입니다. 근거 없는 답변은 낮은 점수를 받습니다.
- 단, 법령DB에 해당 정보가 없을 수 있으므로, 검색 결과가 없어도 합리적 답변이면 인정합니다.
- quality_score 3점 이상이면 통과, 2점 이하면 재검색을 권장합니다."""),
        ("human", """## 사용자 질문
{query}

## 검색된 문서 (요약)
{context_summary}

## 생성된 답변
{answer}

위 답변을 평가해주세요.""")
    ])

    def evaluate_answer(state: AgentState) -> AgentState:
        """답변 평가 노드: 생성된 답변의 품질을 LLM으로 평가"""
        query = state["user_query"]
        answer = state.get("generated_answer", "")
        docs = state.get("retrieved_docs", [])
        retry_count = state.get("retry_count", 0) or 0

        print(f"🔍 [답변 평가 중...] (시도 {retry_count + 1}회)")

        # 검색된 문서 요약 생성
        if docs:
            context_summary = "\n".join([
                f"- {doc.metadata.get('law_name', '문서')} {doc.metadata.get('article_no', '')}: {doc.page_content[:100]}..."
                for doc in docs[:5]
            ])
        else:
            context_summary = "(검색된 문서 없음)"

        # LLM으로 평가
        chain = evaluate_prompt | structured_llm
        evaluation: AnswerEvaluation = chain.invoke({
            "query": query,
            "context_summary": context_summary,
            "answer": answer
        })

        print(f"📊 [평가 결과]")
        print(f"   법적 근거: {'✅' if evaluation.has_legal_basis else '❌'}")
        print(f"   문서 인용: {'✅' if evaluation.cites_retrieved_docs else '❌'}")
        print(f"   답변 적합: {'✅' if evaluation.is_relevant else '❌'}")
        print(
            f"   품질 점수: {'⭐' * evaluation.quality_score} ({evaluation.quality_score}/5)")
        if evaluation.needs_more_search:
            print(f"   ⚠️  추가 검색 권장: {evaluation.improvement_suggestion}")

        return {
            "evaluation_result": evaluation.model_dump(),
            "retry_count": retry_count + 1
        }

    return evaluate_answer


def route_after_evaluation(state: AgentState) -> Literal["search", "end"]:
    """평가 후 라우팅: 재검색 필요 여부에 따라 분기"""
    evaluation = state.get("evaluation_result", {})
    retry_count = state.get("retry_count", 0) or 0

    needs_more_search = evaluation.get("needs_more_search", False)
    quality_score = evaluation.get("quality_score", 3)

    # 무한 루프 방지: 최대 1회만 재시도
    if retry_count >= 2:
        print(f"⚠️  [최대 재시도 횟수 도달] 현재 답변 사용")
        return "end"

    # 품질 점수가 2점 이하이고 추가 검색이 필요하면 재검색
    if needs_more_search and quality_score <= 2:
        print(f"🔄 [재검색 결정] 품질 향상을 위해 다시 검색합니다")
        return "search"

    print(f"✅ [평가 통과] 답변 품질 충분")
    return "end"


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


# ===========================
# 사전 준비 영역: 리소스 초기화
# ===========================
def initialize_resources():
    """임베딩 모델, 벡터스토어, BM25 Retriever 초기화"""

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

    # 5. BM25 Retriever 초기화
    print(f"\n📝 BM25 Retriever 초기화 중...")
    bm25_retriever = None
    try:
        # 전체 문서 수 확인
        collection_info = client.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count
        print(f"   컬렉션 내 전체 문서: {total_points}개")

        # 문서 로드 (BM25용 - 최대 2000개 샘플링)
        sample_size = min(2000, total_points)
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
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
                    metadata={k: v for k, v in payload.items() if k != "text"}
                )
                bm25_docs.append(doc)

        if bm25_docs:
            bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=15)
            print(f"✅ BM25 Retriever 초기화 완료 ({len(bm25_docs)}개 문서 인덱싱)")
        else:
            print("⚠️  BM25용 문서가 없습니다. Vector Search만 사용됩니다.")

    except Exception as e:
        print(f"⚠️  BM25 Retriever 초기화 실패: {e}")
        print("   Vector Search만 사용됩니다.")

    return {
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever
    }


# ===========================
# LangGraph 초기화
# ===========================
def initialize_langgraph_chatbot():
    """LangGraph 기반 RAG 챗봇 초기화 (조건부 분기 포함, 하이브리드 검색 + Query Expansion)"""

    # 사전 준비: 리소스 초기화
    resources = initialize_resources()
    vectorstore = resources["vectorstore"]
    bm25_retriever = resources.get("bm25_retriever")  # BM25 Retriever

    # LLM 설정
    print(f"\n🤖 LLM 설정 중...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        streaming=True
    )
    print("✅ LLM 설정 완료")

    # Query Expander 생성
    print(f"\n🔄 Query Expander 초기화 중...")
    query_expander = create_query_expander(llm)
    print("✅ Query Expander 초기화 완료")

    # 노드 생성
    print(f"\n⚙️  LangGraph 노드 생성 중...")
    analyze_node = create_analyze_query_node(llm)
    clarify_node = create_clarify_node(llm)
    # 하이브리드 검색 노드 (Vector + BM25 + Query Expansion + 유사도 필터링)
    search_node = create_search_node(
        vectorstore, bm25_retriever, query_expander)
    generate_node = create_generate_node(llm)
    # [V5] 답변 평가 노드 (Generator-Critic Light)
    evaluate_node = create_evaluate_node(llm)
    print("✅ 노드 생성 완료 (5개: analyze → search → generate → evaluate)")

    # StateGraph 구성
    print(f"\n🔗 LangGraph 워크플로우 구성 중...")
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("search", search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)

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

    # 검색 후 → 답변 생성 (판례 웹검색 제거, 바로 generate로)
    workflow.add_edge("search", "generate")

    # [V5] 답변 생성 후 → 평가 (Generator-Critic Light)
    workflow.add_edge("generate", "evaluate")

    # [V5] 조건부 분기 2: 평가 후 → 재검색 필요? → search / END
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "search": "search",
            "end": END
        }
    )

    # 그래프 컴파일
    graph = workflow.compile()
    print("✅ LangGraph 구성 완료 (Generator-Critic Light 포함)")

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
        print("🚀 A-TEAM 법률 RAG 챗봇 (LangGraph V5) 초기화 시작")
        print("="*60 + "\n")

        graph = initialize_langgraph_chatbot()

        print("\n" + "="*60)
        print("✅ 🤖 A-TEAM 법률 챗봇 준비 완료!")
        print("="*60)
        print("\n💡 사용 방법:")
        print("  - 노동법, 형사법, 민사법 관련 질문에 응답합니다.")
        print("  - 질문이 모호하면 구체화를 요청합니다.")
        print("  - 'exit', 'quit', '종료'를 입력하면 종료됩니다")
        print("\n📊 워크플로우 (V5 - Generator-Critic Light):")
        print("  ┌─ 질문 분석 ─┬─ [모호함] → 명확화 요청 → 종료")
        print("  │            └─ [명확함] → 하이브리드 검색 → 답변 생성 → 평가")
        print("  │                           (유사도 0.3 미만 필터링)     ↓")
        print(
            "  │                                           [품질 부족] ↓   ↓ [통과]")
        print("  └─────────────────────────────────── 재검색 ←───┘   종료")
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
                    "generated_answer": None,
                    "next_action": None,
                    "evaluation_result": None,
                    "retry_count": 0
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
