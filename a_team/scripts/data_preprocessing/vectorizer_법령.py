"""
법령 데이터 Qdrant 벡터 DB 업로드 스크립트 (Unified / BGE-M3)
- common.vector_db.LegalVectorDB 사용
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

# ============================================================
# 설정
# ============================================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / '..' / 'data'
PROCESSED_FILE = DATA_DIR / 'processed' / 'fd_법령_chunked.json'

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


def main():
    print("=" * 60)
    print("⚖️  법령 데이터 Qdrant 업로드 (Hybrid: Qwen + BGE-M3)")
    print("=" * 60)

    if not PROCESSED_FILE.exists():
        print(f"❌ 전처리된 파일이 없습니다: {PROCESSED_FILE}")
        print("💡 먼저 'uv run a_team/scripts/preprocesser_법령.py'를 실행하세요.")
        return

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


if __name__ == "__main__":
    main()
