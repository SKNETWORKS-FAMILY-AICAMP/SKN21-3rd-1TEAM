"""
Qdrant Cloud의 데이터를 기반으로 RAG 챗봇 평가를 위한 Golden Set 생성 스크립트 (Ragas 0.4.x)

요구사항(커스텀):
1) 총 20개
2) 노동법 10, 민사법 5, 형사법 5
3) 각 분야별 난이도 비율 고급:중급:초급 = 2:1:1 (정수화는 반올림 후 보정)
4) 노동법 질문은 법령 외 문서도 참고하여 답변할 수 있도록 질문 생성(가능하면 해당 플래그 true인 질문을 우선 선택)

구현 방식:
- RAGAS로 분야별로 충분히 큰 풀을 생성
- LLM으로 (분야/난이도/노동-비법령참고가능) 라벨링
- 쿼터에 맞춰 샘플링
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

import pandas as pd
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -----------------------------
# 유틸: 쿼터 계산 (2:1:1 비율)
# -----------------------------
def compute_difficulty_quota(n: int) -> Dict[str, int]:
    """
    고급:중급:초급 = 2:1:1 비율을 n개에 맞게 정수로 할당.
    - round 후 총합 보정 방식.
    반환 키: {"고급": x, "중급": y, "초급": z}
    """
    ratio = {"고급": 2, "중급": 1, "초급": 1}
    total = sum(ratio.values())
    raw = {k: n * v / total for k, v in ratio.items()}
    q = {k: int(round(val)) for k, val in raw.items()}

    # 총합 보정
    diff = n - sum(q.values())
    # diff>0이면 가장 큰 비율(고급)부터 +, diff<0이면 가장 큰 것부터 -
    order = ["고급", "중급", "초급"]
    i = 0
    while diff != 0:
        k = order[i % len(order)]
        if diff > 0:
            q[k] += 1
            diff -= 1
        else:
            if q[k] > 0:
                q[k] -= 1
                diff += 1
        i += 1
    return q


def normalize_domain_label(s: str) -> str:
    s = (s or "").strip()
    if "노동" in s:
        return "노동법"
    if "민사" in s:
        return "민사법"
    if "형사" in s:
        return "형사법"
    return "기타"


def normalize_level_label(s: str) -> str:
    s = (s or "").strip()
    if "고급" in s:
        return "고급"
    if "중급" in s:
        return "중급"
    if "초급" in s:
        return "초급"
    return "중급"


def parse_label_line(line: str) -> Tuple[str, str, bool]:
    """
    기대 형식: "분야|난이도|노동-비법령참고가능(yes/no)"
    예: "노동법|고급|yes"
    """
    parts = [p.strip() for p in (line or "").split("|")]
    if len(parts) < 3:
        return ("기타", "중급", False)
    domain = normalize_domain_label(parts[0])
    level = normalize_level_label(parts[1])
    ns = parts[2].lower()
    non_statute_ok = ns in ("yes", "y", "true", "1", "가능")
    return (domain, level, non_statute_ok)


def build_labeler_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=0)


def label_rows(df: pd.DataFrame, llm: ChatOpenAI) -> pd.DataFrame:
    """
    각 row에 대해 (domain, difficulty, labor_non_statute_ok) 라벨을 부여.
    RAGAS DF 컬럼이 버전에 따라 다를 수 있어 넓게 대응.
    """
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 법률 QA 평가 데이터 라벨러다.\n"
         "입력(질문/정답/컨텍스트 일부)을 보고 다음을 판정한다:\n"
         "1) 분야: 노동법/민사법/형사법/기타\n"
         "2) 난이도: 초급/중급/고급\n"
         "3) (노동법인 경우) 법령 조문만으로 답하기보다, 지침/실무자료/서식/행정해석/가이드/사내규정 등\n"
         "   '법령 외 문서' 참고가 유리한 질문이면 yes, 아니면 no\n\n"
         "출력은 반드시 한 줄로만, 다음 형식:\n"
         "분야|난이도|yes/no\n"
         "예: 노동법|고급|yes"),
        ("human",
         "질문:\n{q}\n\n정답(참고):\n{a}\n\n컨텍스트(발췌):\n{ctx}")
    ])

    def get_col(row, *names):
        for n in names:
            if n in row and pd.notna(row[n]):
                return row[n]
        return ""

    domains = []
    levels = []
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
        domain, level, non_statute_ok = parse_label_line(out)

        domains.append(domain)
        levels.append(level)
        non_statutes.append(bool(non_statute_ok))

    df = df.copy()
    df["domain"] = domains
    df["difficulty"] = levels
    df["labor_non_statute_ok"] = non_statutes
    return df


def sample_with_quota(df: pd.DataFrame, domain_targets: Dict[str, int]) -> pd.DataFrame:
    """
    domain_targets 예: {"노동법":10, "민사법":5, "형사법":5}
    각 도메인 내부에서 난이도 쿼터(2:1:1)를 계산해서 충족하도록 샘플링.
    노동법은 labor_non_statute_ok == True 를 우선 채택(가능하면).
    """
    picked_frames = []

    for domain, n in domain_targets.items():
        dq = compute_difficulty_quota(n)
        domain_df = df[df["domain"] == domain].copy()

        # 노동법이면 non_statute_ok 우선순위 부여
        if domain == "노동법":
            domain_df["__priority"] = domain_df["labor_non_statute_ok"].apply(lambda x: 0 if x else 1)
            domain_df = domain_df.sort_values(["__priority"])

        for level, k in dq.items():
            sub = domain_df[domain_df["difficulty"] == level]
            if len(sub) < k:
                raise ValueError(f"쿼터 충족 실패: {domain}/{level} 필요 {k}개, 보유 {len(sub)}개")
            picked_frames.append(sub.sample(n=k, random_state=42))

    out = pd.concat(picked_frames, ignore_index=True)
    if len(out) != sum(domain_targets.values()):
        raise ValueError("샘플링 결과 개수가 목표와 다릅니다.")
    return out


# --------------------------------
# Qdrant에서 문서 로드(기존 유지)
# --------------------------------
def load_documents_from_qdrant(
    collection_name: str = None,
    limit: int = 0
) -> list:
    from langchain_core.documents import Document
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "A-TEAM")

    print(f"📂 Qdrant DB에서 문서 로드 중...")
    print(f"   Collection: {collection}")

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

    documents = []
    offset = None
    batch_size = 100

    while True:
        results = client.scroll(
            collection_name=collection,
            limit=batch_size,
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
                documents.append(doc)

        offset = next_offset
        if limit > 0 and len(documents) >= limit:
            documents = documents[:limit]
            break
        if next_offset is None:
            break

    print(f"\n📄 총 {len(documents)}개 청킹된 문서 로드 완료\n")
    return documents


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
    from ragas.run_config import RunConfig

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


def split_docs_by_domain(docs: list) -> Dict[str, list]:
    """
    Qdrant payload metadata 기반으로 대략적인 분야별 문서 분리.
    - metadata.category 또는 law_name/source 등에 '노동/민사/형사' 포함 여부로 분류
    """
    buckets = {"노동법": [], "민사법": [], "형사법": [], "기타": []}

    def guess_domain(doc) -> str:
        meta = doc.metadata or {}
        cat = str(meta.get("category", ""))
        law = str(meta.get("law_name", ""))
        src = str(meta.get("source", ""))
        hay = f"{cat} {law} {src}"

        if re.search(r"노동|근로|임금|해고|퇴직", hay):
            return "노동법"
        if re.search(r"민사|계약|손해배상|채권|소유", hay):
            return "민사법"
        if re.search(r"형사|범죄|수사|형벌|공소", hay):
            return "형사법"
        return "기타"

    for d in docs:
        buckets[guess_domain(d)].append(d)
    return buckets


def make_labor_mixed_docs(labor_docs: list, max_docs: int) -> list:
    """
    노동법 문서 중 '법령 외 문서'가 섞이도록 간단히 믹스.
    - law_name이 비어있거나 chunk_type이 법령 청크가 아닌 것들을 non-statute로 간주
    """
    statutes = []
    non_statutes = []
    for d in labor_docs:
        meta = d.metadata or {}
        law_name = str(meta.get("law_name", "")).strip()
        chunk_type = str(meta.get("chunk_type", "")).strip().lower()

        if law_name and ("law" in chunk_type or chunk_type in ("law", "statute", "조문", "법령")):
            statutes.append(d)
        elif law_name:
            # law_name은 있는데 chunk_type이 애매하면 일단 statute로 분류
            statutes.append(d)
        else:
            non_statutes.append(d)

    # 70% statute + 30% non-statute 목표(가능한 만큼)
    target_non = int(max_docs * 0.3)
    target_stat = max_docs - target_non

    picked = []
    if statutes:
        picked += statutes[:min(len(statutes), target_stat)]
    if non_statutes:
        picked += non_statutes[:min(len(non_statutes), target_non)]

    # 부족하면 나머지로 채움
    if len(picked) < max_docs:
        rest = [d for d in labor_docs if d not in picked]
        picked += rest[: (max_docs - len(picked))]

    return picked[:max_docs]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS 기반 Golden Set 생성(커스텀 쿼터/난이도)")
    parser.add_argument('--collection', type=str, default=None)
    parser.add_argument('--sample-size', type=int, default=0, help="Qdrant에서 가져올 문서 수(0=전체)")
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

    # 1) 문서 로드
    all_docs = load_documents_from_qdrant(collection_name=args.collection, limit=args.sample_size)
    if not all_docs:
        print("❌ 로드된 문서가 없습니다. Qdrant 연결을 확인해주세요.")
        return

    # 2) 분야별 문서 버킷
    buckets = split_docs_by_domain(all_docs)
    for k, v in buckets.items():
        print(f"   📌 {k}: {len(v)} docs")

    # 목표 쿼터
    domain_targets = {"노동법": 10, "민사법": 5, "형사법": 5}

    # 3) RAGAS generator 설정
    generator = setup_generator(args.model)

    # 4) 분야별로 풀 생성(충분히 크게)
    frames = []
    for domain, target_n in domain_targets.items():
        docs = buckets.get(domain, [])
        if not docs:
            raise ValueError(f"'{domain}' 문서가 Qdrant에서 발견되지 않았습니다. category/metadata를 확인하세요.")

        # 노동법은 비법령 문서 섞이도록 믹스
        if domain == "노동법":
            docs_for_gen = make_labor_mixed_docs(docs, max_docs=min(len(docs), 300))
        else:
            docs_for_gen = docs[:min(len(docs), 300)]

        pool_n = target_n * max(2, args.pool_mult)
        print(f"\n🧪 [{domain}] 풀 생성: {pool_n}개 (목표 {target_n})")
        df_pool = generate_testset(generator, docs_for_gen, test_size=pool_n)
        df_pool["__generated_domain_hint"] = domain
        frames.append(df_pool)

    df_all = pd.concat(frames, ignore_index=True)
    print(f"\n✅ 전체 풀 생성 완료: {len(df_all)} rows")

    # 5) 라벨링
    labeler = build_labeler_llm(args.model)
    print("\n🏷️  라벨링 중(분야/난이도/노동-비법령참고)...")
    df_labeled = label_rows(df_all, labeler)

    # 6) 쿼터 샘플링(부족하면 에러로 알림)
    print("\n🎯 쿼터 샘플링 중...")
    df_selected = sample_with_quota(df_labeled, domain_targets)

    # 7) 저장
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output
    df_selected.to_json(output_path, orient='records', force_ascii=False, indent=2)

    print(f"\n💾 저장 완료: {output_path}")
    print("\n📊 최종 분포:")
    print(df_selected.groupby(["domain", "difficulty"]).size().to_string())
    print("\n(노동법) 비법령참고가능 개수:", int(df_selected[df_selected["domain"] == "노동법"]["labor_non_statute_ok"].sum()))


if __name__ == '__main__':
    main()