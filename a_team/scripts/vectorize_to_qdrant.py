"""
법령 및 행정해석 데이터 Qdrant 벡터 DB 저장 스크립트
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
import re
from typing import List, Dict, Any
from datetime import datetime

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

COLLECTIONS = {
    'labor_laws': '노동법',
    'civil_laws': '민사법',
    'criminal_laws': '형사법',
    'moel_interpretations': '행정해석'
}


# ============================================================
# 텍스트 전처리 함수
# ============================================================
def clean_law_content(text: str) -> str:
    """법령 조문 텍스트 정리"""
    if not text:
        return ""

    # 개정 이력 태그 간소화: <개정 2021. 1. 5.> -> [개정 2021.1.5]
    text = re.sub(r'<개정\s*([^>]+)>', r'[개정 \1]', text)

    # 연속 공백/줄바꿈 정규화
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def clean_interpretation_text(text: str) -> str:
    """행정해석 텍스트 정리"""
    if not text:
        return ""

    # "목록" 텍스트 제거
    text = re.sub(r'\n*목록$', '', text)

    # 불릿 포인트 정규화
    text = re.sub(r'^[•○◦▶►◇◆■□▪▫·]\s*', '• ', text, flags=re.MULTILINE)

    # 연속 줄바꿈 정규화
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ============================================================
# 청킹 함수
# ============================================================
def chunk_law_data(law_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    법령 데이터를 하이브리드 청킹 전략으로 처리

    전략:
    - 조 단위 기본 청킹 (의미적 완결성 보장)
    - 500자 초과 조문은 항 단위로 분리 (토큰 제한 고려)
    - 항 단위 분리 시 조 컨텍스트 prefix 추가
    """
    chunks = []

    # meta_info에서 법률 정보 추출
    meta = law_data.get('meta_info', {})
    law_name = meta.get('law_name', '')
    law_id = meta.get('law_id', '')
    category = meta.get('category', '')
    url = meta.get('url', '')

    # body에서 조문 처리
    for article in law_data.get('body', []):
        article_no = article.get('article_no', '')
        article_title = article.get('article_title', '')
        article_text = clean_law_content(article.get('article_text_full', ''))
        paragraphs = article.get('paragraphs', [])

        if not article_text or len(article_text) < 10:
            continue

        base_metadata = {
            'source': 'law',
            'law_name': law_name,
            'law_id': law_id,
            'category': category,
            'article_no': article_no,
            'article_title': article_title,
            'url': url
        }

        # 하이브리드 청킹: 조가 짧으면 통째로, 길면 항 단위로 분리
        if len(article_text) <= 500 or len(paragraphs) <= 1:
            # 조 단위 청킹 (짧은 조문 또는 항이 1개뿐인 경우)
            chunks.append({
                'text': article_text,
                'metadata': {
                    **base_metadata,
                    'paragraph_no': '',
                    'chunk_type': 'full_article',
                    'chunk_index': 0
                }
            })
        else:
            # 항 단위 청킹 (긴 조문)
            # 조 제목을 prefix로 붙여 컨텍스트 유지
            prefix = f"{law_name} {article_title}\n\n"

            for i, para in enumerate(paragraphs):
                para_no = para.get('no', str(i + 1))
                para_content = clean_law_content(para.get('content', ''))

                if not para_content or len(para_content) < 10:
                    continue

                # prefix + 항 내용
                chunk_text = prefix + para_content

                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **base_metadata,
                        'paragraph_no': para_no,
                        'chunk_type': 'paragraph',
                        'chunk_index': i
                    }
                })

    return chunks


def chunk_interpretation_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """행정해석 데이터를 Q&A 쌍 단위로 청킹"""
    chunks = []
    parsed = data.get('parsed', {})

    if not parsed.get('parse_success', False):
        # 파싱 실패한 경우 raw_content 사용
        raw = data.get('raw_content', '')
        if raw and len(raw) > 50:
            chunk = {
                'text': clean_interpretation_text(raw),
                'metadata': {
                    'source': 'moel_interpretation',
                    'title': data.get('title', ''),
                    'department': data.get('department', ''),
                    'reg_date': data.get('reg_date', ''),
                    'url': data.get('url', ''),
                    'qa_index': 0,
                    'parse_success': False
                }
            }
            chunks.append(chunk)
        return chunks

    questions = parsed.get('questions', [])
    answers = parsed.get('answers', [])

    # Q&A 쌍 매칭
    for i, (q, a) in enumerate(zip(questions, answers)):
        q_clean = clean_interpretation_text(q)
        a_clean = clean_interpretation_text(a)

        if not q_clean or not a_clean:
            continue

        # 질의-회신 형식으로 결합
        combined_text = f"[질의]\n{q_clean}\n\n[회신]\n{a_clean}"

        chunk = {
            'text': combined_text,
            'metadata': {
                'source': 'moel_interpretation',
                'title': data.get('title', ''),
                'department': data.get('department', ''),
                'reg_date': data.get('reg_date', ''),
                'url': data.get('url', ''),
                'qa_index': i + 1,
                'parse_success': True
            }
        }
        chunks.append(chunk)

    return chunks


# ============================================================
# Qdrant 클라이언트 클래스
# ============================================================
class LegalVectorDB:
    def __init__(self, path: str = None, host: str = None, port: int = 6333, url: str = None, api_key: str = None):
        """
        Qdrant 클라이언트 초기화
        path: 로컬 저장 경로 (None이면 서버 모드)
        host: 서버 호스트 (예: 'localhost')
        port: 서버 포트 (기본 6333)
        url: 클라우드 URL (host 대신 사용 가능)
        api_key: 클라우드 접속용 API 키
        """
        if url:
            # 클라우드 접속 (API Key 필수)
            print(f"Qdrant 클라우드 연결: {url}")
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        elif host:
            # 로컬/서버 접속
            self.client = QdrantClient(host=host, port=port, timeout=60)
            print(f"Qdrant 서버 연결: {host}:{port}")
        elif path:
            # 로컬 파일 모드
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

        print(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("모델 로딩 완료")

    def create_collection(self, name: str, recreate: bool = False):
        """컬렉션 생성"""
        collections = [
            c.name for c in self.client.get_collections().collections]

        if name in collections:
            if recreate:
                print(f"컬렉션 '{name}' 삭제 후 재생성")
                self.client.delete_collection(name)
            else:
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

    def add_documents(self, collection_name: str, chunks: List[Dict[str, Any]], batch_size: int = 8):
        """문서 추가 (메모리 효율적 배치 처리)"""
        if not chunks:
            print(f"추가할 문서 없음")
            return

        print(f"총 {len(chunks)}개 문서 임베딩 중...")

        total_saved = 0
        point_id = 0

        # 작은 배치로 나눠서 처리
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            # 텍스트 추출 (512자로 제한)
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
                        'text': chunk['text'][:2000],
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
            print(f"\r저장됨: {total_saved}/{len(chunks)}", end='', flush=True)

        print(f"\n'{collection_name}'에 {total_saved}개 문서 저장 완료")

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


def process_laws(db: LegalVectorDB, category: str, collection_name: str):
    """법령 데이터 처리"""
    filepath = os.path.join(RAW_DIR, f"{category}.json")

    if not os.path.exists(filepath):
        print(f"파일 없음: {filepath}")
        return

    print(f"\n=== {category} 처리 중 ===")
    data = load_json(filepath)

    all_chunks = []
    for law in data:
        chunks = chunk_law_data(law)
        all_chunks.extend(chunks)

    print(f"총 {len(data)}개 법률, {len(all_chunks)}개 조문")

    db.create_collection(collection_name, recreate=True)
    db.add_documents(collection_name, all_chunks)


def process_interpretations(db: LegalVectorDB):
    """행정해석 데이터 처리"""
    # 가장 최근 파일 찾기
    files = [f for f in os.listdir(PROCESSED_DIR) if f.startswith(
        '행정해석_') and f.endswith('.json')]

    if not files:
        print("행정해석 파일 없음")
        return

    latest_file = sorted(files)[-1]
    filepath = os.path.join(PROCESSED_DIR, latest_file)

    print(f"\n=== 행정해석 처리 중 ({latest_file}) ===")
    data = load_json(filepath)

    all_chunks = []
    for item in data:
        chunks = chunk_interpretation_data(item)
        all_chunks.extend(chunks)

    print(f"총 {len(data)}개 행정해석, {len(all_chunks)}개 Q&A")

    db.create_collection('moel_interpretations', recreate=True)
    db.add_documents('moel_interpretations', all_chunks)


def main():
    print("=" * 60)
    print("법령/행정해석 Qdrant 벡터 DB 구축 (Docker 서버)")
    print("=" * 60)

    # Qdrant 연결 (클라우드/로컬)
    # 로컬 Docker 대신 Cloud URL 사용
    cloud_url = "https://75daa0f4-de48-4954-857a-1fbc276e298f.us-east4-0.gcp.cloud.qdrant.io/"
    api_key = os.getenv("QDRANT_API_KEY")

    if cloud_url and api_key:
        db = LegalVectorDB(url=cloud_url, api_key=api_key)
    else:
        # API 키 없으면 로컬로 폴백
        print("⚠️ QDRANT_API_KEY가 없어 로컬 Docker로 연결합니다.")
        db = LegalVectorDB(host='localhost', port=6333)

    # A-TEAM 컬렉션 생성 (단일 컬렉션에 모든 데이터)
    db.create_collection('A-TEAM', recreate=True)

    # 모든 데이터 수집
    all_chunks = []

    # 1. 법령 데이터 처리
    for category in ['노동법', '민사법', '형사법']:
        filepath = os.path.join(RAW_DIR, f"rd_{category}.json")
        if not os.path.exists(filepath):
            print(f"파일 없음: {filepath}")
            continue

        print(f"\n=== {category} 처리 중 ===")
        data = load_json(filepath)

        for law in data:
            chunks = chunk_law_data(law)
            all_chunks.extend(chunks)

        print(f"현재까지 {len(all_chunks)}개 조문")

    # 2. 행정해석 처리
    files = [f for f in os.listdir(
        PROCESSED_DIR) if '행정해석' in f and f.endswith('.json')]
    if files:
        latest_file = sorted(files)[-1]
        filepath = os.path.join(PROCESSED_DIR, latest_file)
        print(f"\n=== 행정해석 처리 중 ({latest_file}) ===")
        data = load_json(filepath)

        for item in data:
            chunks = chunk_interpretation_data(item)
            all_chunks.extend(chunks)

        print(f"총 {len(all_chunks)}개 문서")

    # 3. A-TEAM 컬렉션에 저장
    print(f"\n=== A-TEAM 컬렉션에 저장 ===")
    db.add_documents('A-TEAM', all_chunks)

    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("=== 저장 완료 ===")
    print("=" * 60)

    info = db.get_collection_info('A-TEAM')
    print(f"• A-TEAM: {info['points_count']}개 문서")

    # 5. 테스트 검색
    print("\n=== 테스트 검색: '퇴직금 중간정산' ===")
    try:
        results = db.search('A-TEAM', '퇴직금 중간정산', top_k=3)
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
