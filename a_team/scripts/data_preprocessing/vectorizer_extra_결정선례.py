"""
결정선례 데이터만 Qdrant 벡터 DB에 추가하는 스크립트
기존 데이터는 유지하고 결정선례만 추가합니다.
"""

from sentence_transformers import SentenceTransformer
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)
import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# 환경변수 로드
load_dotenv()

# ============================================================
# 경로 설정
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# ============================================================
# 설정
# ============================================================
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """긴 텍스트를 일정 크기로 청킹 (오버랩 포함)"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            for sep in ['\n\n', '\n', '. ', '。']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.5:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else end

    return chunks


def load_json(filepath: str) -> Any:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("결정선례 Qdrant 벡터 DB 추가")
    print("=" * 60)

    # Qdrant Cloud 연결
    cloud_url = "https://75daa0f4-de48-4954-857a-1fbc276e298f.us-east4-0.gcp.cloud.qdrant.io/"
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    if not api_key:
        raise ValueError("❌ QDRANT_API_KEY 환경변수가 설정되지 않았습니다.")

    if not collection_name:
        raise ValueError("❌ QDRANT_COLLECTION_NAME 환경변수가 설정되지 않았습니다.")

    print(f"Qdrant 클라우드 연결: {cloud_url}")
    client = QdrantClient(url=cloud_url, api_key=api_key, timeout=60)

    # 현재 컬렉션의 최대 point ID 조회
    print(f"\n컬렉션 '{collection_name}' 정보 조회 중...")
    collection_info = client.get_collection(collection_name)
    current_points = collection_info.points_count or 0
    print(f"현재 저장된 포인트 수: {current_points}")

    # 임베딩 모델 로딩
    print(f"\n임베딩 모델 로딩: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("모델 로딩 완료")

    # 결정선례 데이터 로드
    decision_file = os.path.join(PROCESSED_DIR, "fd_법령외_결정선례.json")
    if not os.path.exists(decision_file):
        raise FileNotFoundError(f"파일 없음: {decision_file}")

    print(f"\n=== 결정선례 데이터 로드 중 ===")
    documents = load_json(decision_file)
    print(f"결정선례 {len(documents)}개 문서 로드 완료")

    # 청킹
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

    print(f"청킹 완료: {len(documents)}개 문서 → {len(all_chunks)}개 청크")

    # 배치 임베딩 및 업로드
    batch_size = 8
    total_saved = 0
    point_id = current_points  # 기존 포인트 이후부터 시작

    print(f"\n=== '{collection_name}' 컬렉션에 저장 ===")
    print(f"시작 point ID: {point_id}")

    for batch_start in range(0, len(all_chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(all_chunks))
        batch_chunks = all_chunks[batch_start:batch_end]

        # 텍스트 추출 (임베딩용 - 512자 제한)
        texts = [c['text'][:512] for c in batch_chunks]

        # 배치 임베딩
        embeddings = model.encode(texts, show_progress_bar=False)

        # 포인트 생성
        points = []
        for chunk, embedding in zip(batch_chunks, embeddings):
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    'text': chunk['text'][:2000],
                    **chunk['metadata']
                }
            )
            points.append(point)
            point_id += 1

        # Qdrant에 업서트
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        total_saved += len(points)
        print(f"\r저장됨: {total_saved}/{len(all_chunks)}", end='', flush=True)

    print(f"\n\n'{collection_name}'에 {total_saved}개 청크 저장 완료")

    # 결과 요약
    print("\n" + "=" * 60)
    print("=== 저장 완료 ===")
    print("=" * 60)

    final_info = client.get_collection(collection_name)
    print(f"• 최종 포인트 수: {final_info.points_count}")
    print(f"• 추가된 포인트 수: {total_saved}")

    # 테스트 검색
    print("\n=== 테스트 검색: '고용보험' ===")
    try:
        query_embedding = model.encode("고용보험")
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=3
        )
        for i, hit in enumerate(results.points, 1):
            print(f"\n[{i}] 점수: {hit.score:.3f}")
            print(f"    소스: {hit.payload.get('source', '')}")
            print(f"    제목: {hit.payload.get('title', '')}")
            print(f"    내용: {hit.payload.get('text', '')[:100]}...")
    except Exception as e:
        print(f"검색 실패: {e}")


if __name__ == '__main__':
    main()
