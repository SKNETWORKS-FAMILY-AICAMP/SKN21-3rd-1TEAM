"""
법령 데이터 Qdrant 벡터 DB 업로드 스크립트 (Unified / BGE-M3)
- common.vector_db.LegalVectorDB 사용
<<<<<<< HEAD
- 로컬 또는 클라우드 저장소 선택 가능
"""
import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Common Module Import (Fix: 3 levels up to reach project root)
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))

from a_team.scripts.common.vector_db import LegalVectorDB  # noqa: E402 # isort: skip

=======
"""
from a_team.scripts.common.vector_db import LegalVectorDB
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Common Module Import (Fix: 3 levels up to reach project root)
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))
>>>>>>> 209151e353aba59a2423f8158163afcb4a0cdf48

# ============================================================
# 설정
# ============================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / '..' / 'data'
PROCESSED_FILE = DATA_DIR / 'processed' / 'fd_법령_chunked.json'
LOCAL_QDRANT_PATH = DATA_DIR / 'qdrant_local'

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPARSE_MODEL = "BAAI/bge-m3"

# 환경 변수 로드 (Project Root .env)
# scripts/data_preprocessing/../../.. -> Project Root
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
ENV_PATH = PROJECT_ROOT / '.env'
if not ENV_PATH.exists():
    # Try finding it relative to current working dir if script assumption fails
    ENV_PATH = Path(os.getcwd()) / '.env'

print(f"🌍 Loading .env from: {ENV_PATH}")
load_dotenv(ENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "A-TEAM")


def load_json(filepath):
    """JSON 파일 로드"""
    print(f"📂 Loading: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
<<<<<<< HEAD


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="법령 데이터를 Qdrant에 업로드 (Hybrid Search)"
    )
    parser.add_argument(
        '--storage-mode',
        type=str,
        choices=['local', 'cloud', 'server'],
        default='cloud',
        help='Qdrant 저장소 모드 선택 (local: 로컬 디스크, cloud: Qdrant Cloud, server: Docker 서버)'
    )
    parser.add_argument(
        '--local-path',
        type=str,
        default=str(LOCAL_QDRANT_PATH),
        help='로컬 저장소 경로 (storage-mode=local일 때 사용)'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default=COLLECTION_NAME,
        help='Qdrant 컬렉션 이름'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='기존 컬렉션 삭제 후 재생성'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=12,
        help='업로드 배치 크기'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='MPS 대신 CPU 사용 (느리지만 메모리 안정적)'
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='시작 청크 인덱스 (병렬 처리용)'
    )
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='종료 청크 인덱스 (병렬 처리용, None=끝까지)'
    )
    return parser.parse_args()
=======
>>>>>>> 209151e353aba59a2423f8158163afcb4a0cdf48


def main():
    args = parse_args()

    print("=" * 60)
    print("⚖️  법령 데이터 Qdrant 업로드 (Hybrid: Qwen + BGE-M3)")
<<<<<<< HEAD
    print(f"📦 저장소 모드: {args.storage_mode.upper()}")
=======
>>>>>>> 209151e353aba59a2423f8158163afcb4a0cdf48
    print("=" * 60)

    if not PROCESSED_FILE.exists():
        print(f"❌ 전처리된 파일이 없습니다: {PROCESSED_FILE}")
        print("💡 먼저 'uv run a_team/scripts/data_preprocessing/preprocesser_법령.py'를 실행하세요.")
        return

<<<<<<< HEAD
    # 스트리밍 방식으로 변경 (메모리 효율: ~1GB → ~50MB)
    from a_team.scripts.common.json_utils import stream_json_array, count_json_array_items

    print("📊 청크 수 확인 중...")
    total_chunks = count_json_array_items(PROCESSED_FILE)
    print(f"📊 총 청크: {total_chunks:,}개 (스트리밍 모드)")

    # DB 초기화 (저장소 모드에 따라)
    if args.storage_mode == 'local':
        print(f"💾 로컬 저장소 사용: {args.local_path}")
        db = LegalVectorDB(
            local_path=args.local_path,
            dense_model_name=EMBEDDING_MODEL,
            sparse_model_name=SPARSE_MODEL,
            force_cpu=args.force_cpu
        )
    elif args.storage_mode == 'cloud':
        print(f"🌐 클라우드 저장소 사용")
        if not QDRANT_URL or not QDRANT_API_KEY:
            print("❌ .env 파일에 QDRANT_URL과 QDRANT_API_KEY를 설정하세요.")
            return
        db = LegalVectorDB(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            dense_model_name=EMBEDDING_MODEL,
            sparse_model_name=SPARSE_MODEL,
            force_cpu=args.force_cpu
        )
    else:  # server
        print(f"🏠 서버 모드 사용 (localhost:6333)")
        db = LegalVectorDB(
            host='localhost',
            port=6333,
            dense_model_name=EMBEDDING_MODEL,
            sparse_model_name=SPARSE_MODEL,
            force_cpu=args.force_cpu
        )

    # 컬렉션 생성
    db.create_collection(args.collection_name, recreate=args.recreate)

    # 이어서 업로드 (현재 저장된 청크 수 확인)
    if not args.recreate:
        info = db.get_collection_info(args.collection_name)
        start_idx = info['points_count']
        if start_idx > 0:
            print(
                f"\n🔄 이어서 업로드: {start_idx:,}개 이미 저장됨, {total_chunks - start_idx:,}개 남음")
        else:
            print(f"\n🆕 새로운 업로드 시작")
        start_id = start_idx
    else:
        start_id = 0
        start_idx = 0

    if start_idx >= total_chunks:
        print("✅ 모든 청크가 이미 업로드되었습니다!")
        return

    # 배치 스트리밍으로 처리 (메모리 효율: 5000개씩 로드)
    print(f"\n🚀 배치 스트리밍 업로드 (업로드 배치: {args.batch_size}, 메모리 배치: 5000)...")

    processed_count = 0
    for batch_chunks in stream_json_array(PROCESSED_FILE, batch_size=5000):
        # 이미 처리된 청크 건너뛰기
        if processed_count + len(batch_chunks) <= start_idx:
            processed_count += len(batch_chunks)
            continue

        # 부분적으로 처리된 배치 처리
        if processed_count < start_idx:
            skip_count = start_idx - processed_count
            batch_chunks = batch_chunks[skip_count:]
            processed_count = start_idx

        # 업로드
        current_start_id = start_id + (processed_count - start_idx)
        db.upsert_chunks(args.collection_name, batch_chunks,
                         batch_size=args.batch_size,
                         start_id=current_start_id)

        processed_count += len(batch_chunks)

        # 메모리 정리
        del batch_chunks
        import gc
        gc.collect()

    print("\n" + "=" * 60)
    print(f"✅ 완료! 컬렉션 '{args.collection_name}'에 총 {total_chunks:,}개 청크 저장됨")
    if args.storage_mode == 'local':
        print(f"📂 저장 위치: {args.local_path}")
    print("=" * 60)
=======
    chunks = load_json(PROCESSED_FILE)
    print(f"📊 로드된 청크: {len(chunks)}개")

    # DB 초기화
    db = LegalVectorDB(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        dense_model_name=EMBEDDING_MODEL,
        sparse_model_name=SPARSE_MODEL
    )

    # 컬렉션 생성 (Main Script -> recreate=True)
    db.create_collection(COLLECTION_NAME, recreate=True)

    # 업서트
    db.upsert_chunks(COLLECTION_NAME, chunks, batch_size=12)
>>>>>>> 209151e353aba59a2423f8158163afcb4a0cdf48


if __name__ == "__main__":
    main()
