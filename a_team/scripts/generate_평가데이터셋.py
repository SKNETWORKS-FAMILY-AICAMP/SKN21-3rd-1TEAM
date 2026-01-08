"""
Qdrant Cloud의 데이터를 기반으로 RAG 챗봇 평가를 위한 Golden Set 생성 스크립트 (Ragas 0.4.x)

요구사항(커스텀):
1) 총 20개
2) 노동법 10, 민사법 5, 형사법 5
3) 노동법 질문은 법령 외 문서도 참고하여 답변할 수 있도록 질문 생성(가능하면 해당 플래그 true인 질문을 우선 선택)

구현 방식:
- RAGAS로 분야별로 충분히 큰 풀을 생성
- LLM으로 (분야/노동-비법령참고가능) 라벨링
- 분야별 목표 개수에 맞춰 샘플링
"""

import argparse
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

# .env 파일에서 환경변수 로드
load_dotenv()

# LangSmith 트레이싱 명시적 비활성화 (토큰 절약)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

warnings.filterwarnings("ignore", category=DeprecationWarning)


def normalize_domain_label(s: str) -> str:
    s = (s or "").strip()
    if "노동" in s:
        return "노동법"
    if "민사" in s:
        return "민사법"
    if "형사" in s:
        return "형사법"
    return "기타"


def parse_label_line(line: str) -> Tuple[str, bool]:
    """
    기대 형식: "분야|노동-비법령참고가능(yes/no)"
    예: "노동법|yes"
    """
    parts = [p.strip() for p in (line or "").split("|")]
    if len(parts) < 2:
        return ("기타", False)
    domain = normalize_domain_label(parts[0])
    ns = parts[1].lower()
    non_statute_ok = ns in ("yes", "y", "true", "1", "가능")
    return (domain, non_statute_ok)


def build_labeler_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=0)


def reformat_answers(df: pd.DataFrame, llm: ChatOpenAI) -> pd.DataFrame:
    """
    RAGAS가 생성한 답변(reference)을 원하는 템플릿 형식으로 재작성.
    템플릿:
    - "질문에 대한 답변: ..."
    - "관련 법령 조항: ..."
    - "추가 설명: ..."
    """
    prompt = ChatPromptTemplate.from_template("""너는 법률 QA 데이터셋의 답변을 정해진 템플릿으로 재작성하는 역할이다.
주어진 질문과 원본 답변을 참고하여, 아래 템플릿 형식으로 재작성해라.
원본 답변의 내용을 충실히 반영하되, 템플릿에 맞게 구조화해라.

### 템플릿:
- "질문에 대한 답변: (핵심 답변 1~2문장)"
- "관련 법령 조항: (법령명 및 조항 번호. 여러 개일 수 있음)"
- "추가 설명: (보충 설명, 예외사항, 주의점 등. 2~4문장)"

### 입력:
질문: {question}

원본 답변:
{original_answer}

### 출력:
템플릿 형식으로 재작성된 답변만 출력해라. 다른 설명은 하지 마라.""")

    def get_col(row, *names):
        for n in names:
            if n in row and pd.notna(row[n]):
                return row[n]
        return ""

    new_answers = []
    for _, row in df.iterrows():
        question = get_col(row, "user_input", "question")
        original = get_col(row, "reference", "ground_truth", "answer")

        if not original or not question:
            new_answers.append(original)
            continue

        chain = prompt | llm
        result = chain.invoke({
            "question": str(question)[:1000],
            "original_answer": str(original)[:2000]
        }).content.strip()
        new_answers.append(result)

    df = df.copy()
    # reference 컴럼 이름 확인 후 업데이트
    if "reference" in df.columns:
        df["reference"] = new_answers
    elif "ground_truth" in df.columns:
        df["ground_truth"] = new_answers
    elif "answer" in df.columns:
        df["answer"] = new_answers
    else:
        df["reference"] = new_answers
    
    return df


def label_rows(df: pd.DataFrame, llm: ChatOpenAI) -> pd.DataFrame:
    """
    각 row에 대해 (domain, labor_non_statute_ok) 라벨을 부여.
    RAGAS DF 컬럼이 버전에 따라 다를 수 있어 넓게 대응.
    """

    prompt = ChatPromptTemplate.from_template("""너는 법률 QA 평가 데이터 라벨러다.
입력(질문/정답/컨텍스트 일부)을 보고 다음을 판정한다:

1) 분야: 노동법/민사법/형사법/기타
2) (노동법인 경우) 법령 조문만으로 답하기 어렵고, 행정해석/판례/Q&A/판정선례 등
   '법령 외 문서' 참고가 필요한 질문이면 yes, 아니면 no

출력은 반드시 한 줄로만, 다음 형식:
분야|yes/no
예: 노동법|yes

---
질문:
{q}

정답(참고):
{a}

컨텍스트(발췌):
{ctx}""")

    def get_col(row, *names):
        for n in names:
            if n in row and pd.notna(row[n]):
                return row[n]
        return ""

    domains = []
    non_statutes = []

    for _, row in df.iterrows():
        q = get_col(row, "user_input", "question")
        a = get_col(row, "reference", "ground_truth", "answer")
        ctx_val = get_col(row, "contexts", "context")
        # contexts가 list일 수도 있어서 텍스트로 축약
        if isinstance(ctx_val, list):
            ctx = "\n---\n".join([str(x) for x in ctx_val[:3]])
        else:
            ctx = str(ctx_val)[:1500]

        chain = prompt | llm
        out = chain.invoke({"q": str(q)[:800], "a": str(a)[:1200], "ctx": ctx}).content.strip()
        domain, non_statute_ok = parse_label_line(out)

        domains.append(domain)
        non_statutes.append(bool(non_statute_ok))

    df = df.copy()
    df["domain"] = domains
    df["labor_non_statute_ok"] = non_statutes
    return df


def sample_with_quota(df: pd.DataFrame, domain_targets: Dict[str, int]) -> pd.DataFrame:
    """
    domain_targets 예: {"노동법":10, "민사법":5, "형사법":5}
    각 도메인별 목표 개수만큼 샘플링.
    노동법은 labor_non_statute_ok == True 를 우선 채택(가능하면).
    """
    picked_frames = []

    for domain, n in domain_targets.items():
        domain_df = df[df["domain"] == domain].copy()

        if len(domain_df) == 0:
            print(f"⚠️  [{domain}] 해당 분야 질문이 없습니다. 스킵합니다.")
            continue

        # 노동법이면 non_statute_ok 우선순위 부여
        if domain == "노동법":
            domain_df["__priority"] = domain_df["labor_non_statute_ok"].apply(lambda x: 0 if x else 1)
            domain_df = domain_df.sort_values(["__priority"])

        # 목표 개수만큼 샘플링
        take = min(len(domain_df), n)
        if take < n:
            print(f"⚠️  [{domain}] 필요 {n}개, 보유 {len(domain_df)}개 → {take}개 샘플링")
        
        sampled = domain_df.head(take)
        picked_frames.append(sampled)

    if not picked_frames:
        raise ValueError("샘플링된 데이터가 없습니다. 생성된 풀을 확인해주세요.")
    
    out = pd.concat(picked_frames, ignore_index=True)
    
    expected = sum(domain_targets.values())
    if len(out) != expected:
        print(f"⚠️  최종 샘플 수: {len(out)}개 (목표 {expected}개)")
    
    return out


# --------------------------------
# Qdrant에서 문서 로드 (최적화 버전)
# --------------------------------
def load_documents_from_qdrant_by_domain(
    collection_name: str = None,
    docs_per_domain: int = 500
) -> Dict[str, list]:
    """
    Qdrant에서 분야별로 필요한 문서만 샘플링해서 로드.
    전체 스캔 대신 랜덤 샘플링으로 빠르게 가져옴.
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "A-TEAM")

    print(f"📂 Qdrant DB에서 분야별 문서 샘플링 중...")
    print(f"   Collection: {collection}")
    print(f"   분야별 최대: {docs_per_domain}개")

    if qdrant_url and qdrant_api_key:
        print(f"   URL: {qdrant_url[:30]}...")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )
    else:
        print("   Local Docker: localhost:6333")
        client = QdrantClient(host="localhost", port=6333)

    try:
        collection_info = client.get_collection(collection_name=collection)
        total_points = collection_info.points_count
        print(f"   총 포인트 수: {total_points}")
    except Exception as e:
        raise ConnectionError(f"Qdrant 컬렉션 '{collection}'에 연결할 수 없습니다: {e}")

    buckets = {"노동법": [], "노동법_법령외": [], "민사법": [], "형사법": [], "기타": []}
    
    # 다양성 확보를 위한 분산 샘플링
    sample_size = min(docs_per_domain * 6, 5000)
    batch_size = 20  # 작은 배치로 다양한 위치에서 샘플링
    num_sampling_points = max(100, sample_size // batch_size)  # 최소 100개 위치
    
    print(f"\n   📥 분산 랜덤 샘플링 중 ({sample_size}개 목표, {num_sampling_points}개 위치)")
    
    # 전체 범위를 균등 분할 후 각 구간에서 랜덤 샘플링
    all_sampled = []
    
    if total_points > sample_size:
        # 전체 범위를 num_sampling_points개 구간으로 나누기
        segment_size = total_points // num_sampling_points
        
        for i in range(num_sampling_points):
            if len(all_sampled) >= sample_size:
                break
            
            # 각 구간 내에서 랜덤 오프셋 선택
            segment_start = i * segment_size
            segment_end = min((i + 1) * segment_size, total_points)
            
            if segment_end - segment_start < batch_size:
                random_offset = segment_start
            else:
                random_offset = random.randint(segment_start, max(segment_start, segment_end - batch_size))
            
            try:
                results = client.scroll(
                    collection_name=collection,
                    limit=batch_size,
                    offset=random_offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, _ = results
                
                for point in points:
                    if len(all_sampled) >= sample_size:
                        break
                    
                    payload = point.payload or {}
                    text = payload.get("text", "")
                    
                    if text and len(text) > 30:
                        doc = Document(
                            page_content=text,
                            metadata={
                                "id": str(point.id),
                                "source": payload.get("source", ""),
                                "law_name": payload.get("law_name", ""),
                                "law_id": payload.get("law_id", ""),
                                "article_no": payload.get("article_no", ""),
                                "article_title": payload.get("article_title", ""),
                                "paragraph_no": payload.get("paragraph_no", ""),
                                "chunk_type": payload.get("chunk_type", ""),
                                "category": payload.get("category", ""),
                                "chunk_index": payload.get("chunk_index", 0),
                            }
                        )
                        all_sampled.append(doc)
            except Exception as e:
                # 일부 오프셋에서 오류 발생 시 스킵
                continue
    else:
        # 전체 데이터가 샘플 사이즈보다 작으면 전체 로드
        offset = None
        while True:
            results = client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = results
            if not points:
                break
                
            for point in points:
                payload = point.payload or {}
                text = payload.get("text", "")
                
                if text and len(text) > 30:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "id": str(point.id),
                            "source": payload.get("source", ""),
                            "law_name": payload.get("law_name", ""),
                            "law_id": payload.get("law_id", ""),
                            "article_no": payload.get("article_no", ""),
                            "article_title": payload.get("article_title", ""),
                            "paragraph_no": payload.get("paragraph_no", ""),
                            "chunk_type": payload.get("chunk_type", ""),
                            "category": payload.get("category", ""),
                            "chunk_index": payload.get("chunk_index", 0),
                        }
                    )
                    all_sampled.append(doc)
            
            offset = next_offset
            if offset is None:
                break

    print(f"   ✅ {len(all_sampled)}개 문서 샘플링 완료")

    # 분야별 분류
    # 법령 외 문서의 source 값들 (모두 노동법 관련)
    # interpretation: 행정해석, case_law: 주요판정사례, moel_qa: 고용노동부QA, 판정선례: 결정선례
    non_statute_sources = {"interpretation", "case_law", "moel_qa", "판정선례"}
    
    def classify_domain(doc) -> str:
        meta = doc.metadata or {}
        # 1. 법령 여부: law_name(또는 law_id)이 있으면 법령
        if meta.get("law_name") or meta.get("law_id"):
            category = str(meta.get("category", ""))
            if category in buckets:
                return category
            return "기타"
        # 2. 법령 외 문서: source로 분류
        src = str(meta.get("source", ""))
        if src in non_statute_sources:
            return "노동법_법령외"
        return "기타"

    # 셔플해서 순서 랜덤화 (같은 법률이 연속으로 오는 것 방지)
    random.shuffle(all_sampled)
    
    for doc in all_sampled:
        domain = classify_domain(doc)
        if len(buckets[domain]) < docs_per_domain:
            buckets[domain].append(doc)
    
    # 각 분야 내에서도 다시 셔플 (법률명 기준 다양성 확보)
    for domain in buckets:
        random.shuffle(buckets[domain])

    print(f"\n📄 분야별 로드 완료:")
    for k, v in buckets.items():
        if v:
            # 해당 분야의 문서 다양성 체크 (법령 + 법령외)
            law_names = set(doc.metadata.get("law_name") for doc in v if doc.metadata.get("law_name"))
            sources = set(doc.metadata.get("source", "") for doc in v)
            non_statute_count = sum(1 for doc in v if not doc.metadata.get("law_name"))
            print(f"   📌 {k}: {len(v)} docs (법령 {len(law_names)}종류, 법령외 {non_statute_count}개)")
        else:
            print(f"   📌 {k}: {len(v)} docs")

    return buckets


# ---------------------------------------------------------
# [수정] 온도를 강제로 1로 고정하는 커스텀 LLM 클래스
# ---------------------------------------------------------
class ForceTemperature1ChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if 'temperature' in kwargs:
            kwargs['temperature'] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if 'temperature' in kwargs:
            kwargs['temperature'] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def setup_generator(model_name: str = "gpt-4o-mini") -> TestsetGenerator:
    print(f"🤖 LLM 설정 중(생성): {model_name}")

    generator_llm = ForceTemperature1ChatOpenAI(
        model=model_name,
        temperature=1,
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings
    )

    print("✅ TestsetGenerator 설정 완료\n")
    return generator


def generate_testset(generator: TestsetGenerator, documents: list, test_size: int) -> pd.DataFrame:
    run_config = RunConfig(
        max_retries=3,
        max_wait=60,
        max_workers=4,
        timeout=120,
        exception_types=(Exception,),
    )

    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=test_size,
        raise_exceptions=False,
        run_config=run_config,
    )
    return testset.to_pandas()


def main():
    parser = argparse.ArgumentParser(description="RAGAS 기반 Golden Set 생성(커스텀 쿼터/난이도)")
    parser.add_argument('--collection', type=str, default=None)
    parser.add_argument('--docs-per-domain', type=int, default=500, help="분야별 샘플링할 문서 수(기본 500)")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='생성/라벨링에 사용할 LLM 모델')
    parser.add_argument('--output', type=str, default='golden_set_quota_20.json')
    parser.add_argument('--pool-mult', type=int, default=6, help="분야별 생성 풀 크기 배수(기본 6배)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수를 설정해주세요.")
        return

    print("=" * 60)
    print("🏛️  RAG 평가용 Golden Set 생성 (RAGAS + 쿼터 샘플링)")
    print("=" * 60)

    # 1) 분야별 문서 직접 샘플링 (최적화)
    buckets = load_documents_from_qdrant_by_domain(
        collection_name=args.collection, 
        docs_per_domain=args.docs_per_domain
    )
    
    total_docs = sum(len(v) for v in buckets.values())
    if total_docs == 0:
        print("❌ 로드된 문서가 없습니다. Qdrant 연결을 확인해주세요.")
        return

    # 목표 쿼터
    domain_targets = {"노동법": 10, "민사법": 5, "형사법": 5}

    # 2) RAGAS generator 설정
    generator = setup_generator(args.model)

    # 3) 분야별로 풀 생성(충분히 크게)
    frames = []
    for domain, target_n in domain_targets.items():
        # 노동법은 법령 + 법령외 합쳐서 사용
        if domain == "노동법":
            labor_law_docs = buckets.get("노동법", [])
            labor_extra_docs = buckets.get("노동법_법령외", [])
            docs = labor_law_docs + labor_extra_docs
        else:
            docs = buckets.get(domain, [])
        
        if not docs:
            print(f"⚠️  '{domain}' 문서가 없습니다. 스킵합니다.")
            continue

        # 노동법은 법령 + 법령외 문서를 함께 제공 (RAGAS가 자연스럽게 질문 생성)
        if domain == "노동법":
            all_labor_docs = labor_law_docs + labor_extra_docs
            random.shuffle(all_labor_docs)
            docs_for_gen = all_labor_docs[:min(len(all_labor_docs), 300)]
        else:
            docs_for_gen = docs[:min(len(docs), 300)]

        pool_n = target_n * max(2, args.pool_mult)
        print(f"\n🧪 [{domain}] 풀 생성: {pool_n}개 (목표 {target_n})")
        df_pool = generate_testset(generator, docs_for_gen, test_size=pool_n)
        df_pool["__generated_domain_hint"] = domain
        frames.append(df_pool)

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n✅ 전체 풀 생성 완료: {len(df_all)} rows")

    # 5) 답변 템플릿 재작성
    labeler = build_labeler_llm(args.model)
    print("\n📝 답변 템플릿 재작성 중...")
    df_all = reformat_answers(df_all, labeler)

    # 6) 라벨링
    print("\n🏷️  라벨링 중(분야/노동-비법령참고)...")
    df_labeled = label_rows(df_all, labeler)

    # 7) 분야별 샘플링
    print("\n🎯 샘플링 중...")
    df_selected = sample_with_quota(df_labeled, domain_targets)

    # 8) 저장
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output
    df_selected.to_json(output_path, orient='records', force_ascii=False, indent=2)

    print(f"\n💾 저장 완료: {output_path}")
    print("\n📊 최종 분포:")
    print(df_selected["domain"].value_counts().to_string())
    print("\n(노동법) 비법령참고가능 개수:", int(df_selected[df_selected["domain"] == "노동법"]["labor_non_statute_ok"].sum()))


if __name__ == '__main__':
    main()