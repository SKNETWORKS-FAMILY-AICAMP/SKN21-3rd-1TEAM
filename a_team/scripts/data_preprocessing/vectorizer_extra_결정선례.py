"""
결정선례 데이터 Qdrant 업로드 스크립트 (Unified / BGE-M3)
"""
from a_team.scripts.common.vector_db import LegalVectorDB
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Common Module Import (Fix: 3 levels up)
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))

# ============================================================
# 경로 설정
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SPARSE_MODEL = "BAAI/bge-m3"

# 환경 변수 로드 (Project Root .env)
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent.parent
ENV_PATH = PROJECT_ROOT / '.env'
if not ENV_PATH.exists():
    ENV_PATH = Path(os.getcwd()) / '.env'
print(f"🌍 Loading .env from: {ENV_PATH}")
load_dotenv(ENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "A-TEAM")


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """텍스트 청킹"""
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap
        if start >= len(text) - overlap:
            break

    return chunks


def main():
    print("=" * 60)
    print("⚖️  결정선례 데이터 Qdrant 추가 업로드")
    print("=" * 60)

    decision_file = os.path.join(PROCESSED_DIR, "fd_법령외_결정선례.json")
    if not os.path.exists(decision_file):
        raise FileNotFoundError(f"파일 없음: {decision_file}")

    # DB 초기화
    db = LegalVectorDB(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        dense_model_name=EMBEDDING_MODEL,
        sparse_model_name=SPARSE_MODEL
    )

    # Collection check
    db.create_collection(COLLECTION_NAME, recreate=False)

    # Start ID
    info = db.get_collection_info(COLLECTION_NAME)
    start_id = info['points_count']
    print(f"Current Points: {start_id}")

    print(f"\n=== 결정선례 데이터 로드 중 ===")
    documents = load_json(decision_file)
    print(f"결정선례 {len(documents)}개 문서 로드 완료")

    print(f"\n총 {len(documents)}개 문서 청킹 중...")
    all_chunks = []
    for doc_idx, doc in enumerate(documents):
        text = doc['text']
        metadata = doc['metadata']

        text_chunks = chunk_text(text, chunk_size=800, overlap=100)

        for chunk_idx, chunk_str in enumerate(text_chunks):
            chunk_metadata = {
                **metadata,
                'parent_doc_id': doc_idx,
                'chunk_index': chunk_idx,
                'total_chunks': len(text_chunks),
                'chunk_length': len(chunk_str)
            }
            all_chunks.append({
                'text': chunk_str,
                'metadata': chunk_metadata
            })

    print(f"청킹 완료: {len(all_chunks)}개 청크 생성됨")

    # 업로드
    db.upsert_chunks(COLLECTION_NAME, all_chunks,
                     batch_size=12, start_id=start_id)


if __name__ == "__main__":
    main()
