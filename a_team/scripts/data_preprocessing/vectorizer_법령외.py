"""
주요판정사례, 행정해석 데이터를 Qdrant 벡터 DB에 저장하는 스크립트
"""

from sentence_transformers import SentenceTransformer
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    models
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
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# ============================================================
# 설정
# ============================================================
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Qwen3 임베딩 모델
EMBEDDING_DIM = 1024  # Qwen3 차원


# ============================================================
# 청킹 함수
# ============================================================
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    긴 텍스트를 일정 크기로 청킹 (오버랩 포함)

    Args:
        text: 청킹할 텍스트
        chunk_size: 청크 크기 (문자 수)
        overlap: 청크 간 오버랩 크기

    Returns:
        청크 리스트
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # 마지막 청크가 아니면 문장 경계에서 자르기 시도
        if end < len(text):
            # 줄바꿈이나 마침표 찾기
            for sep in ['\n\n', '\n', '. ', '。']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.5:  # 최소 50% 이상은 채워야 함
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 오버랩 적용
        start = end - overlap if end < len(text) else end

    return chunks


# ============================================================
# Qdrant 클라이언트 클래스
# ============================================================
class LegalVectorDB:
    def __init__(self, url: str, api_key: str):
        """
        Qdrant Cloud 클라이언트 초기화

        Args:
            url: Qdrant Cloud URL
            api_key: Qdrant Cloud API 키
        """
        print(f"Qdrant 클라우드 연결: {url}")
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)

        print(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("모델 로딩 완료")

    def create_collection(self, name: str):
        """컬렉션 생성 (없을 경우만)"""
        collections = [
            c.name for c in self.client.get_collections().collections]

        if name in collections:
            print(f"컬렉션 '{name}' 이미 존재")
            return

        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"컬렉션 '{name}' 생성 완료")

    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]], batch_size: int = 8):
        """
        문서 추가 (청킹 + 메모리 효율적 배치 처리)

        Args:
            collection_name: 컬렉션 이름
            documents: 문서 리스트 (각 문서는 {'text': str, 'metadata': dict} 형태)
            batch_size: 배치 크기
        """
        if not documents:
            print(f"추가할 문서 없음")
            return

        print(f"총 {len(documents)}개 문서 청킹 및 임베딩 중...")

        # 1단계: 모든 문서를 청크로 분할
        all_chunks = []
        for doc_idx, doc in enumerate(documents):
            text = doc['text']
            metadata = doc['metadata']

            # 텍스트를 청크로 분할
            text_chunks = chunk_text(text, chunk_size=800, overlap=100)

            # 각 청크에 메타데이터 추가
            for chunk_idx, chunk_str in enumerate(text_chunks):
                chunk_metadata = {
                    **metadata,
                    'parent_doc_id': doc_idx,  # 원본 문서 ID
                    'chunk_index': chunk_idx,   # 청크 순서
                    'total_chunks': len(text_chunks),  # 전체 청크 수
                    'chunk_length': len(chunk_str)
                }
                all_chunks.append({
                    'text': chunk_str,
                    'metadata': chunk_metadata
                })

        print(f"청킹 완료: {len(documents)}개 문서 → {len(all_chunks)}개 청크")

        # 2단계: 배치 임베딩 및 업로드
        total_saved = 0
        point_id = 0

        for batch_start in range(0, len(all_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(all_chunks))
            batch_chunks = all_chunks[batch_start:batch_end]

            # 텍스트 추출 (임베딩용 - 512자 제한)
            texts = [c['text'][:512] for c in batch_chunks]

            # 배치 임베딩
            embeddings = self.model.encode(texts, show_progress_bar=False)

            # 포인트 생성
            points = []
            for chunk, embedding in zip(batch_chunks, embeddings):
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        'text': chunk['text'][:2000],  # 전체 텍스트 저장 (2000자 제한)
                        **chunk['metadata']
                    }
                )
                points.append(point)
                point_id += 1

            # Qdrant에 업서트
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            total_saved += len(points)
            print(f"\r저장됨: {total_saved}/{len(all_chunks)}",
                  end='', flush=True)

        print(f"\n'{collection_name}'에 {total_saved}개 청크 저장 완료")

    def search(self, collection_name: str, query: str, top_k: int = 5) -> List[Dict]:
        """유사 문서 검색"""
        query_embedding = self.model.encode(query)

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=top_k
        )

        return [
            {
                'score': hit.score,
                'text': hit.payload.get('text', ''),
                'metadata': {k: v for k, v in hit.payload.items() if k != 'text'}
            }
            for hit in results.points
        ]

    def get_collection_info(self, name: str) -> Dict:
        """컬렉션 정보 조회"""
        info = self.client.get_collection(name)
        return {
            'name': name,
            'vectors_count': info.vectors_count if hasattr(info, 'vectors_count') else 0,
            'points_count': info.points_count if hasattr(info, 'points_count') else 0
        }


# ============================================================
# 메인 실행 함수
# ============================================================
def load_json(filepath: str) -> Any:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("주요판정사례/행정해석 Qdrant 벡터 DB 구축")
    print("=" * 60)

    # Qdrant Cloud 연결
    cloud_url = "https://75daa0f4-de48-4954-857a-1fbc276e298f.us-east4-0.gcp.cloud.qdrant.io/"
    api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    if not api_key:
        raise ValueError("❌ QDRANT_API_KEY 환경변수가 설정되지 않았습니다.")

    if not collection_name:
        raise ValueError("❌ QDRANT_COLLECTION_NAME 환경변수가 설정되지 않았습니다.")

    db = LegalVectorDB(url=cloud_url, api_key=api_key)

    # 컬렉션 생성 (없을 경우만)
    db.create_collection(collection_name)

    # 모든 데이터 수집
    all_chunks = []

    # 1. 주요판정사례 데이터 로드
    case_law_file = os.path.join(PROCESSED_DIR, "fd_법령외_주요판정사례.json")
    if os.path.exists(case_law_file):
        print(f"\n=== 주요판정사례 데이터 로드 중 ===")
        data = load_json(case_law_file)
        all_chunks.extend(data)
        print(f"주요판정사례 {len(data)}개 문서 로드 완료")
    else:
        print(f"파일 없음: {case_law_file}")

    # 2. 행정해석 데이터 로드
    interp_file = os.path.join(PROCESSED_DIR, "fd_법령외_행정해석.json")
    if os.path.exists(interp_file):
        print(f"\n=== 행정해석 데이터 로드 중 ===")
        data = load_json(interp_file)
        all_chunks.extend(data)
        print(f"행정해석 {len(data)}개 문서 로드 완료")
    else:
        print(f"파일 없음: {interp_file}")

    # 3. 고용노동부 Q&A 데이터 로드
    moel_qa_file = os.path.join(PROCESSED_DIR, "fd_법령외_고용노동부QA.json")
    if os.path.exists(moel_qa_file):
        print(f"\n=== 고용노동부 Q&A 데이터 로드 중 ===")
        data = load_json(moel_qa_file)
        all_chunks.extend(data)
        print(f"고용노동부 Q&A {len(data)}개 문서 로드 완료")
    else:
        print(f"파일 없음: {moel_qa_file}")

    # 4. 중앙부처 1차 해석 (판정선례) 데이터 로드
    qa_resp_file = os.path.join(PROCESSED_DIR, "fd_법령외_판정선례.json")
    if os.path.exists(qa_resp_file):
        print(f"\n=== 중앙부처 1차 해석 데이터 로드 중 ===")
        data = load_json(qa_resp_file)
        all_chunks.extend(data)
        print(f"중앙부처 1차 해석 {len(data)}개 문서 로드 완료")
    else:
        print(f"파일 없음: {qa_resp_file}")

    # 5. 결정선례 데이터 로드
    decision_file = os.path.join(PROCESSED_DIR, "fd_법령외_결정선례.json")
    if os.path.exists(decision_file):
        print(f"\n=== 결정선례 데이터 로드 중 ===")
        data = load_json(decision_file)
        all_chunks.extend(data)
        print(f"결정선례 {len(data)}개 문서 로드 완료")
    else:
        print(f"파일 없음: {decision_file}")

    print(f"\n총 {len(all_chunks)}개 문서 로드 완료")

    # 컬렉션에 저장
    print(f"\n=== '{collection_name}' 컬렉션에 저장 ===")
    db.add_documents(collection_name, all_chunks)

    # 결과 요약
    print("\n" + "=" * 60)
    print("=== 저장 완료 ===")
    print("=" * 60)

    info = db.get_collection_info(collection_name)
    print(f"• {collection_name}: {info['points_count']}개 문서")

    # 테스트 검색
    print("\n=== 테스트 검색: '퇴직금 중간정산' ===")
    try:
        results = db.search(collection_name, '퇴직금 중간정산', top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] 점수: {r['score']:.3f}")
            print(
                f"    법률: {r['metadata'].get('law_title', r['metadata'].get('title', ''))}")
            print(f"    조문: {r['metadata'].get('article_num', '')}")
            print(f"    내용: {r['text'][:100]}...")
    except Exception as e:
        print(f"검색 실패: {e}")


if __name__ == '__main__':
    main()
