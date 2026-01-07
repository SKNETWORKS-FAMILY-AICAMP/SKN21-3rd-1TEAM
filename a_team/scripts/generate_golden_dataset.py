"""
노동법 RAG 챗봇 평가용 Golden Set 자동 생성 스크립트 (Ragas 0.4.x)

Ragas TestsetGenerator를 사용하여 PDF/텍스트 문서로부터
다양한 유형의 평가용 질문-답변 쌍을 자동 생성합니다.

Tech Stack:
    - Python 3.10+
    - ragas 0.4.x
    - langchain / langchain-openai / langchain-community
    - pandas

Usage:
    # 기본 실행 (OPENAI_API_KEY 환경변수 필요)
    python generate_golden_dataset.py

    # 생성할 테스트셋 크기 지정
    python generate_golden_dataset.py --test-size 50

    # 데이터 폴더 경로 지정
    python generate_golden_dataset.py --data-dir ./custom_data
"""

from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
import os
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 경고 메시지 필터링 (선택사항)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_documents_from_qdrant(
    collection_name: str = None,
    limit: int = 0
) -> list:
    """
    Qdrant DB에서 청킹된 문서를 직접 가져옵니다.

    실제 벡터 DB에 저장된 청크를 사용하여 Golden Set을 만들면
    Context Precision/Recall 평가가 정확해집니다.

    Args:
        collection_name: Qdrant 컬렉션 이름 (기본값: 환경변수 QDRANT_COLLECTION_NAME)
        limit: 가져올 최대 문서 수 (0이면 전체)

    Returns:
        List of LangChain Document objects
    """
    from langchain_core.documents import Document
    from qdrant_client import QdrantClient

    # 스크립트 디렉토리의 .env 파일 로드 (강제)
    script_dir = Path(__file__).parent
    load_dotenv(script_dir / ".env", override=True)

    # 환경변수에서 Qdrant 설정 가져오기
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = collection_name or os.getenv(
        "QDRANT_COLLECTION_NAME", "A-TEAM")

    print(f"📂 Qdrant DB에서 문서 로드 중...")
    print(f"   Collection: {collection}")

    # Qdrant 클라이언트 초기화
    if qdrant_url and qdrant_api_key:
        print(f"   URL: {qdrant_url[:30]}...")
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )
    else:
        # 로컬 Docker Qdrant
        print("   Local Docker: localhost:6333")
        client = QdrantClient(host="localhost", port=6333)

    # 컬렉션 정보 확인
    try:
        collection_info = client.get_collection(collection_name=collection)
        total_points = collection_info.points_count
        print(f"   총 포인트 수: {total_points}")
    except Exception as e:
        raise ConnectionError(f"Qdrant 컬렉션 '{collection}'에 연결할 수 없습니다: {e}")

    # 문서 가져오기 (스크롤 API 사용)
    # 문서 가져오기 (스크롤 API 사용)
    # 다양성을 위해 여러 구간에서 조금씩 가져오는 전략 사용
    documents = []

    if limit > 0 and total_points > limit:
        # 분할 가져오기 설정
        num_partitions = 10  # 10군데에서 나눠서 가져옴
        limit_per_partition = max(1, limit // num_partitions)

        print(
            f"\n   🎲 다양성 확보를 위해 {num_partitions}개 구간에서 각 {limit_per_partition}개씩 랜덤 샘플링합니다.")

        # 랜덤 시작 위치들 생성 (겹치지 않게 정렬)
        import random
        max_start = max(0, total_points - limit_per_partition - 1)
        start_offsets = sorted([random.randint(0, max_start)
                               for _ in range(num_partitions)])

        for i, start_offset in enumerate(start_offsets):
            # Qdrant에서 해당 위치의 포인트 가져오기
            results = client.scroll(
                collection_name=collection,
                limit=limit_per_partition,
                offset=start_offset,
                with_payload=True,
                with_vectors=False
            )

            points, _ = results

            # 문서 변환 및 추가
            chunk_docs = []
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
                    chunk_docs.append(doc)

            print(
                f"      [{i+1}/{num_partitions}] Offset {start_offset} ~ : {len(chunk_docs)}개 로드")
            documents.extend(chunk_docs)

            # 목표 수량이 채워지면 중단 (혹시 모를 오버헤드 방지)
            if len(documents) >= limit:
                break

        # 리스트가 너무 길어지면 자르기
        if len(documents) > limit:
            documents = documents[:limit]

    else:
        # 전체 가져오기 또는 데이터가 적을 때 (기존 로직 유지)
        offset = None
        batch_size = 100

        while True:
            # Qdrant에서 배치로 포인트 가져오기
            results = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False  # 벡터는 필요 없음
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

            # 다음 배치
            offset = next_offset

            # limit이 지정되어 있고 도달했으면 중단
            if limit > 0 and len(documents) >= limit:
                documents = documents[:limit]
                break

            if next_offset is None:
                break

    # 소스별 통계 출력
    source_counts = {}
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"\n   📊 소스별 문서 수:")
    for src, count in sorted(source_counts.items()):
        print(f"      • {src}: {count}개")

    print(f"\n📄 총 {len(documents)}개 청킹된 문서 로드 완료\n")
    return documents

# ---------------------------------------------------------
# [수정] 온도를 강제로 1로 고정하는 커스텀 LLM 클래스
# (일부 모델이 temperature!=1을 지원하지 않을 때 사용)
# ---------------------------------------------------------


# ---------------------------------------------------------
# [수정] 온도를 1로 고정하고, 한국어 출력을 강제하는 커스텀 LLM
# ---------------------------------------------------------
class KoreanForceChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import HumanMessage

        # 한국어 강제 지침 추가 (마지막에 추가하여 가장 높은 우선순위 부여)
        korean_instruction = "IMPORTANT: You must generate ALL outputs (Questions, Answers, Reasoning, Scenarios) in Korean (한국어). Do not use English."
        messages.append(HumanMessage(content=korean_instruction))

        if 'temperature' in kwargs:
            kwargs['temperature'] = 1

        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import HumanMessage

        korean_instruction = "IMPORTANT: You must generate ALL outputs (Questions, Answers, Reasoning, Scenarios) in Korean (한국어). Do not use English."
        messages.append(HumanMessage(content=korean_instruction))

        if 'temperature' in kwargs:
            kwargs['temperature'] = 1

        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


def setup_generator(model_name: str = "gpt-5.2") -> TestsetGenerator:
    """
    Ragas 0.4.x TestsetGenerator를 설정합니다.

    Args:
        model_name: 사용할 OpenAI 모델명 (gpt-4o, gpt-4-turbo 등)

    Returns:
        설정된 TestsetGenerator 인스턴스
    """
    print(f"🤖 LLM 설정 중: {model_name} (한국어 강제 적용)")

    # ---------------------------------------------------------
    # Generator LLM 설정 (커스텀 래퍼 사용)
    # 입력 문서가 한국어이므로 출력도 한국어로 생성됨
    # ---------------------------------------------------------
    generator_llm = KoreanForceChatOpenAI(
        model=model_name,
        temperature=1,
    )

    # ---------------------------------------------------------
    # Embeddings 설정
    # ---------------------------------------------------------
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    # ---------------------------------------------------------
    # Ragas 0.4.x: TestsetGenerator.from_langchain() 사용
    # [수정] NERExtractor 에러 회피를 위해 transforms를 명시적으로 지정
    # ---------------------------------------------------------
    from ragas.testset.transforms import KeyphraseExtractor, SummaryExtractor

    # 사용할 Transform 정의 (NER 제외)
    # NERExtractor가 Pydantic output parser 에러를 유발하므로 제외함
    transforms = [
        KeyphraseExtractor(llm=generator_llm),
        SummaryExtractor(llm=generator_llm),
    ]

    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings
    )

    # [중요] Default transforms를 커스텀 transforms로 교체
    generator.knowledge_graph.transforms = transforms

    print("✅ TestsetGenerator 설정 완료\n")
    return generator


def generate_testset(
    generator: TestsetGenerator,
    documents: list,
    test_size: int = 30
) -> pd.DataFrame:
    """
    문서로부터 테스트셋을 생성합니다.

    Args:
        generator: 설정된 TestsetGenerator
        documents: 로드된 문서 리스트
        test_size: 생성할 질문 개수

    Returns:
        생성된 테스트셋 DataFrame
    """
    from ragas.run_config import RunConfig

    print(f"📝 테스트셋 생성 중 (목표: {test_size}개)")
    print("   노동법 특성상 조건부/추론 질문이 자동으로 많이 생성됩니다.")
    print()

    # ---------------------------------------------------------
    # RunConfig: 에러 시 재시도 및 예외 무시 설정
    # ---------------------------------------------------------
    run_config = RunConfig(
        max_retries=3,           # 실패 시 최대 3회 재시도
        max_wait=60,             # 재시도 간 최대 대기 시간
        max_workers=4,           # 동시 처리 수 제한
        timeout=120,             # 개별 작업 타임아웃
        exception_types=(Exception,),  # 모든 예외 재시도
    )

    # ---------------------------------------------------------
    # Ragas 0.4.x: generate_with_langchain_docs 메서드 사용
    # [수정] distributions 명시
    # ---------------------------------------------------------
    from ragas.testset.evolutions import simple, reasoning, multi_context
    dist = {
        simple: 0.5,
        reasoning: 0.3,
        multi_context: 0.2
    }

    try:
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=test_size,
            distributions=dist,
            raise_exceptions=False,
            run_config=run_config,
        )
    except Exception as e:
        print(f"\n⚠️ 테스트셋 생성 중 에러 발생: {e}")
        print("   일부 문서에서 파싱 실패. 샘플 사이즈를 줄이거나 다시 시도하세요.")
        raise

    # DataFrame으로 변환
    df = testset.to_pandas()

    print(f"\n✅ {len(df)}개 질문-답변 쌍 생성 완료")
    return df


def main():
    """메인 실행 함수"""
    import argparse

    # ---------------------------------------------------------
    # CLI 인자 파싱
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='노동법 RAG 평가용 Golden Set 생성 (Ragas 0.4.x TestsetGenerator)'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default=None,
        help='Qdrant 컬렉션 이름 (기본값: 환경변수 QDRANT_COLLECTION_NAME)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=30,
        help='생성할 테스트 질문 개수 (기본값: 30)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=200,
        help='사용할 문서 샘플링 개수 (0이면 전체 사용, 기본값: 200)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='사용할 LLM 모델 (기본값: gpt-4o-mini)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='labor_law_golden_set.json',
        help='출력 파일명 (기본값: labor_law_golden_set.json)'
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # API 키 확인
    # ---------------------------------------------------------
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수를 설정해주세요.")
        print("   export OPENAI_API_KEY='your-api-key'")
        return

    print("=" * 60)
    print("🏛️  노동법 RAG 평가용 Golden Set 생성기 (Ragas 0.4.x)")
    print("=" * 60)
    print()

    # ---------------------------------------------------------
    # Step 1: 문서 로드
    # ---------------------------------------------------------
    documents = load_documents_from_qdrant(
        collection_name=args.collection,
        limit=args.sample_size
    )

    if not documents:
        print("❌ 로드된 문서가 없습니다. Qdrant 연결을 확인해주세요.")
        return

    # ---------------------------------------------------------
    # Step 2: Generator 설정
    # ---------------------------------------------------------
    generator = setup_generator(args.model)

    # ---------------------------------------------------------
    # Step 3: 테스트셋 생성
    # ---------------------------------------------------------
    df = generate_testset(
        generator=generator,
        documents=documents,
        test_size=args.test_size
    )

    # ---------------------------------------------------------
    # Step 4: 결과 저장
    # ---------------------------------------------------------
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output

    # 기존 파일이 있으면 로드해서 병합
    if output_path.exists():
        try:
            existing_df = pd.read_json(output_path)
            print(f"\n📂 기존 데이터셋 로드: {len(existing_df)}개 샘플")

            # 컬럼 매핑 확인 (기존 데이터와 새 데이터 컬럼이 다를 수 있음)
            # Ragas 버전에 따라 컬럼명이 조금씩 다를 수 있으므로 유연하게 대처
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            print(f"➕ 새 데이터 {len(df)}개 추가 -> 총 {len(combined_df)}개")
            df = combined_df
        except Exception as e:
            print(f"⚠️ 기존 파일 병합 실패 (덮어쓰기 진행): {e}")

    df.to_json(output_path, orient='records', force_ascii=False, indent=2)

    print(f"\n💾 저장 완료: {output_path}")

    # ---------------------------------------------------------
    # Step 5: 결과 미리보기
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 생성된 데이터 미리보기")
    print("=" * 60)
    print(f"\n컬럼: {list(df.columns)}")

    print("\n샘플 질문 3개:")
    for i, row in df.head(3).iterrows():
        q = row.get('user_input', row.get('question', 'N/A'))
        a = row.get('reference', row.get('ground_truth', 'N/A'))
        print(f"\n[{i+1}] Q: {str(q)[:80]}...")
        print(f"    A: {str(a)[:80]}...")


if __name__ == '__main__':
    main()
