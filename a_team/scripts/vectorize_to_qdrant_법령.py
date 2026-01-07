"""
법령 데이터 Qdrant 벡터 DB 업로드 스크립트
- processed/law_chunks.json 로드
- Qdrant 컬렉션(A-TEAM)에 업서트
"""

from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import json
import os
import sys
from typing import List, Dict, Any
from pathlib import Path

# 환경변수 로드
load_dotenv()

# ============================================================
# 설정
# ============================================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'
PROCESSED_FILE = DATA_DIR / 'processed' / 'law_chunks.json'

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "A-TEAM")


# ============================================================
# Qdrant 클라이언트 클래스
# ============================================================
class LegalVectorDB:
    def __init__(self, url: str = None, api_key: str = None, host: str = None, port: int = 6333):
        """Qdrant 클라이언트 초기화"""
        if url and api_key:
            print(f"🌐 Qdrant 클라우드 연결: {url[:30]}...")
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        elif host:
            print(f"🏠 Qdrant 서버 연결: {host}:{port}")
            self.client = QdrantClient(host=host, port=port, timeout=60)
        else:
            print("⚠️ 연결 정보 없음, 메모리 모드")
            self.client = QdrantClient(":memory:")

        print(f"🧠 임베딩 모델 로딩: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def create_collection(self, name: str, recreate: bool = False):
        """컬렉션 생성"""
        collections = [
            c.name for c in self.client.get_collections().collections]
        if name in collections:
            if recreate:
                print(f"♻️  컬렉션 '{name}' 재생성 (삭제 후 생성)")
                self.client.delete_collection(name)
            else:
                print(f"✅ 컬렉션 '{name}' 이미 존재")
                return

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print(f"✨ 컬렉션 '{name}' 생성 완료")

    def upsert_chunks(self, collection_name: str, chunks: List[Dict[str, Any]], batch_size: int = 32):
        """청크 업서트 (배치 처리)"""
        if not chunks:
            print("❌ 업로드할 청크가 없습니다.")
            return

        total = len(chunks)
        print(f"🚀 총 {total}개 청크 업로드 시작...")

        # 포인트 ID 생성을 위한 오프셋 (기존 데이터와 충돌 방지 필요시 조정)
        start_id = 0

        for i in range(0, total, batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c['text'] for c in batch]

            # 임베딩
            embeddings = self.model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True)

            points = []
            for idx, (chunk, vector) in enumerate(zip(batch, embeddings)):
                # 메타데이터에 텍스트 포함 (페이로드 저장용)
                payload = chunk['metadata'].copy()
                payload['text'] = chunk['text']

                points.append(PointStruct(
                    id=start_id + i + idx,
                    vector=vector.tolist(),
                    payload=payload
                ))

            self.client.upsert(collection_name=collection_name, points=points)
            print(
                f"\r📥 저장 중: {i + len(batch)}/{total} ({(i + len(batch))/total*100:.1f}%)", end='', flush=True)

        print(f"\n✅ '{collection_name}' 업로드 완료!")

    def search(self, img_query: str, top_k: int = 3):
        """테스트 검색"""
        vec = self.model.encode(img_query).tolist()
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME, query=vec, limit=top_k).points
        return hits


# ============================================================
# 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print("⚖️  법령 데이터 Qdrant 업로드")
    print("=" * 60)

    # 1. 전처리된 파일 로드
    if not PROCESSED_FILE.exists():
        print(f"❌ 전처리된 파일이 없습니다: {PROCESSED_FILE}")
        print("💡 먼저 'uv run a_team/scripts/preprocesser_법령.py'를 실행하세요.")
        return

    print(f"📂 파일 로드 중: {PROCESSED_FILE}")
    with open(PROCESSED_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"📊 로드된 청크: {len(chunks)}개")

    # 2. Qdrant 연결
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")

    if not url:
        # 로컬 폴백
        db = LegalVectorDB(host='localhost', port=6333)
    else:
        db = LegalVectorDB(url=url, api_key=key)

    # 3. 업로드
    # 주의: recreate=True로 하면 기존 데이터 날라감. 필요시 False로 변경.
    # 하지만 Clean 구축을 위해 True 유지 (사용자 의도에 따라 조정 가능)
    db.create_collection(COLLECTION_NAME, recreate=True)
    db.upsert_chunks(COLLECTION_NAME, chunks)

    # 4. 검증
    print("\n🔍 검색 테스트: '퇴직금 중간정산'")
    hits = db.search("퇴직금 중간정산")
    for i, h in enumerate(hits, 1):
        meta = h.payload
        print(
            f"\n[{i}] {meta.get('law_name')} {meta.get('article_title')} (Score: {h.score:.3f})")
        print(f"    {meta.get('text')[:100]}...")


if __name__ == '__main__':
    main()
