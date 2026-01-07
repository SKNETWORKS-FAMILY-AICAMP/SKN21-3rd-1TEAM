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

import os
import warnings
from pathlib import Path
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

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
    documents = []
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


class ForceTemperature1ChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if 'temperature' in kwargs:
            kwargs['temperature'] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
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
    print(f"🤖 LLM 설정 중: {model_name}")

    # ---------------------------------------------------------
    # Generator LLM 설정 (커스텀 래퍼 사용)
    # 입력 문서가 한국어이므로 출력도 한국어로 생성됨
    # ---------------------------------------------------------
    generator_llm = ForceTemperature1ChatOpenAI(
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
    # ---------------------------------------------------------
    generator = TestsetGenerator.from_langchain(
        llm=generator_llm,
        embedding_model=embeddings
    )

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
    # ---------------------------------------------------------
    try:
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=test_size,
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
        default='gpt-5-mini',
        help='사용할 LLM 모델 (기본값: gpt-5-mini)'
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
