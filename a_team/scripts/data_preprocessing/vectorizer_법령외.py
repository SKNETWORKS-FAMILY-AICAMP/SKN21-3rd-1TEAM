"""
주요판정사례, 행정해석 데이터 Qdrant 업로드 스크립트 (Unified / BGE-M3)
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
    print("⚖️  법령외 데이터(판례/해석) Qdrant 업로드 (Hybrid)")
    print("=" * 60)

    # 타겟 파일들
    targets = [
        "fd_법령외_주요판정사례.json",
        "fd_법령외_행정해석.json",
        "fd_법령외_고용노동부QA.json"
    ]

    # DB 초기화
    db = LegalVectorDB(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        dense_model_name=EMBEDDING_MODEL,
        sparse_model_name=SPARSE_MODEL
    )

    # 컬렉션 확인 (재생성 X)
    db.create_collection(COLLECTION_NAME, recreate=False)

    # 현재 ID 조회 (이어쓰기)
    info = db.get_collection_info(COLLECTION_NAME)
    start_id = info['points_count']
    print(f"Current Points: {start_id}")

    for filename in targets:
        filepath = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ 파일 없음: {filename} (Skip)")
            continue

        print(f"\nProcessing {filename}...")
        documents = load_json(filepath)

        all_chunks = []
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})

            # 청킹
            text_chunks = chunk_text(text)

            for i, chunk_text_str in enumerate(text_chunks):
                chunk_meta = metadata.copy()
                chunk_meta['chunk_index'] = i
                all_chunks.append({
                    'text': chunk_text_str,
                    'metadata': chunk_meta
                })

        print(f"Uploading {len(all_chunks)} chunks for {filename}...")
        db.upsert_chunks(COLLECTION_NAME, all_chunks,
                         batch_size=12, start_id=start_id)

        # ID 업데이트
        start_id += len(all_chunks)


if __name__ == "__main__":
    main()
