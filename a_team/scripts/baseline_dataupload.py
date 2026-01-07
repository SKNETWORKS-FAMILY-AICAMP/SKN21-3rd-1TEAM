import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# 환경 변수 로드
_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH)


def load_labor_law_data(file_path):
    """노동법 데이터 로드 및 전처리"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ❌ 파일 로드 실패 ({file_path.name}): {e}")
        return [], []
    
    documents = []
    metadatas = []
    
    for law in tqdm(data, desc=f"처리 중: {file_path.name}"):
        title = law.get('title', '')
        category = law.get('category', '노동법')
        url = law.get('url', '')
        
        for article in law.get('articles', []):
            article_num = article.get('article_num', '')
            content = article.get('content', '').strip()
            
            if content:
                # 텍스트 구성: [법령명] 조항 번호\n본문
                text = f"[{title}] {article_num}\n{content}"
                documents.append(text)
                metadatas.append({
                    'source': 'labor_law',
                    'title': title,
                    'article_num': article_num,
                    'category': category,
                    'url': url
                })
    
    return documents, metadatas


def load_case_law_data(file_path):
    """판례 데이터 로드 및 전처리"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ❌ 파일 로드 실패 ({file_path.name}): {e}")
        return [], []
    
    documents = []
    metadatas = []
    
    for case in tqdm(data, desc=f"처리 중: {file_path.name}"):
        제목 = case.get('제목', '')
        자료구분 = case.get('자료구분', '')
        판정사항 = case.get('판정사항', '').strip()
        판정요지 = case.get('판정요지', '').strip()
        
        if 판정사항 or 판정요지:
            # 텍스트 구성: [판례] 제목\n판정사항\n판정요지
            text_parts = [f"[판례: {제목}]"]
            if 판정사항:
                text_parts.append(f"판정사항: {판정사항}")
            if 판정요지:
                text_parts.append(f"판정요지: {판정요지}")
            
            text = "\n".join(text_parts)
            documents.append(text)
            metadatas.append({
                'source': 'case_law',
                'title': 제목,
                'category': 자료구분,
                'number': case.get('번호', ''),
                'reg_date': case.get('등록일', '')
            })
    
    return documents, metadatas


def load_interpretation_data(file_path):
    """행정해석 데이터 로드 및 전처리"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ❌ 파일 로드 실패 ({file_path.name}): {e}")
        return [], []
    
    documents = []
    metadatas = []
    
    for item in tqdm(data, desc=f"처리 중: {file_path.name}"):
        title = item.get('title', '').strip()
        url = item.get('url', '')
        department = item.get('department', '')
        
        if title:
            # 텍스트 구성: [행정해석] 제목
            text = f"[행정해석] {title}"
            documents.append(text)
            metadatas.append({
                'source': 'interpretation',
                'title': title,
                'url': url,
                'department': department,
                'number': item.get('number', ''),
                'reg_date': item.get('reg_date', '')
            })
    
    return documents, metadatas


def main():
    """데이터 업로드 메인 함수"""
    
    print("\n" + "="*60)
    print("🚀 법률 데이터 Qdrant 업로드 시작")
    print("="*60 + "\n")
    
    # 1. 임베딩 모델 로드
    print("📥 임베딩 모델 로드 중 (Qwen/Qwen3-Embedding-0.6B)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로드 완료\n")
    
    # 2. 로컬 Qdrant 클라이언트 연결
    print("📡 로컬 Qdrant 연결 중...")
    try:
        client = QdrantClient(host="localhost", port=6333)
        # 연결 테스트
        client.get_collections()
        print("✅ Qdrant 연결 완료\n")
    except Exception as e:
        print(f"❌ Qdrant 연결 실패: {e}")
        print("💡 로컬 Qdrant가 실행 중인지 확인하세요: docker run -p 6333:6333 qdrant/qdrant")
        return
    
    # 3. 컬렉션 확인 및 생성
    collection_name = "A-TEAM-local"
    embedding_dim = 1024  # Qwen3-Embedding-0.6B 실제 차원
    print(f"🔍 컬렉션 확인 중 ({collection_name})...")
    
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        # 기존 컬렉션 정보 확인
        collection_info = client.get_collection(collection_name)
        existing_dim = collection_info.config.params.vectors.size
        
        if existing_dim != embedding_dim:
            print(f"⚠️  차원 불일치 감지 (기존: {existing_dim}, 필요: {embedding_dim})")
            print(f"🗑️  기존 컬렉션 삭제 중...")
            client.delete_collection(collection_name)
            print(f"✅ 컬렉션 삭제 완료")
            collections.remove(collection_name)
        else:
            print(f"✅ 컬렉션 '{collection_name}' 존재 확인 (차원: {existing_dim})")
    
    if collection_name not in collections:
        print(f"🆕 컬렉션 생성 중 (차원: {embedding_dim})...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
        print(f"✅ 컬렉션 '{collection_name}' 생성 완료")
    print()
    
    # 4. 벡터스토어 생성
    print(f"🗂️  벡터스토어 초기화 중...")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text"
    )
    print("✅ 벡터스토어 초기화 완료\n")
    
    # 5. 데이터 파일 경로 설정
    data_dir = Path(__file__).parent / "data" / "raw"
    files = {
        "labor_law": data_dir / "rd_노동법.json",
        "case_law": data_dir / "rd_주요판례.json",
        "interpretation": data_dir / "rd_행정해석.json"
    }
    
    # 6. 데이터 로드
    all_documents = []
    all_metadatas = []
    
    print("📂 데이터 로드 중...\n")
    
    # 노동법 데이터
    if files["labor_law"].exists():
        docs, metas = load_labor_law_data(files["labor_law"])
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        print(f"  ✅ 노동법: {len(docs)}개 문서 로드")
    else:
        print(f"  ⚠️  노동법 파일 없음: {files['labor_law']}")
    
    # 판례 데이터
    if files["case_law"].exists():
        docs, metas = load_case_law_data(files["case_law"])
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        print(f"  ✅ 주요판례: {len(docs)}개 문서 로드")
    else:
        print(f"  ⚠️  판례 파일 없음: {files['case_law']}")
    
    # 행정해석 데이터
    if files["interpretation"].exists():
        docs, metas = load_interpretation_data(files["interpretation"])
        all_documents.extend(docs)
        all_metadatas.extend(metas)
        print(f"  ✅ 행정해석: {len(docs)}개 문서 로드")
    else:
        print(f"  ⚠️  행정해석 파일 없음: {files['interpretation']}")
    
    print(f"\n📊 총 {len(all_documents)}개 문서 로드 완료")
    
    # 7. Qdrant에 업로드
    if all_documents:
        print("\n⬆️  Qdrant에 업로드 중 (임베딩 생성 및 저장)...")
        print("   ⏳ 대량 데이터 처리 중... 시간이 소요될 수 있습니다.\n")
        
        # 배치 단위로 업로드 (메모리 효율) - 작은 배치로 GPU 메모리 부족 방지
        batch_size = 5
        total_batches = (len(all_documents) + batch_size - 1) // batch_size
        
        try:
            for i in range(0, len(all_documents), batch_size):
                batch_docs = all_documents[i:i+batch_size]
                batch_metas = all_metadatas[i:i+batch_size]
                
                vectorstore.add_texts(
                    texts=batch_docs,
                    metadatas=batch_metas
                )
                
                current_batch = i // batch_size + 1
                print(f"   ✅ 배치 {current_batch}/{total_batches} 업로드 완료 ({len(batch_docs)}개 문서)")
            
            print(f"\n✅ 데이터 업로드 완료!")
            print(f"   - 총 {len(all_documents)}개 문서")
            print(f"   - 컬렉션: {collection_name}")
            print(f"   - Qdrant: localhost:6333")
            print("\n" + "="*60)
            print("🎉 업로드 성공! 이제 baseline_local.py로 챗봇을 실행하세요.")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ 업로드 중 오류 발생: {e}")
            print("💡 일부 데이터는 업로드되었을 수 있습니다.")
    else:
        print("\n❌ 업로드할 데이터가 없습니다.")


if __name__ == "__main__":
    main()