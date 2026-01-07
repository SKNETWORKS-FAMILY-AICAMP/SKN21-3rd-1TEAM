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

from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
import os
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# 경고 메시지 필터링 (선택사항)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_documents(data_dir: str) -> list:
    """
    지정된 폴더에서 JSON 파일을 로드합니다.

    사용하는 파일:
    - raw/rd_노동법.json, rd_민사법.json, rd_형사법.json (법령)
    - processed/fd_법령외_고용노동부QA.json (고용노동부 Q&A)
    - processed/fd_법령외_판례.json (판례)
    - processed/fd_법령외_행정해석.json (행정해석)

    Returns:
        List of LangChain Document objects
    """
    from langchain_core.documents import Document
    import json

    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")

    print(f"📂 문서 로드 중: {data_dir}")

    # ==========================================================
    # 1. processed 폴더: {text, metadata} 형식
    # ==========================================================
    processed_files = [
        data_path / "processed" / "fd_법령외_고용노동부QA.json",
        data_path / "processed" / "fd_법령외_판례.json",
        data_path / "processed" / "fd_법령외_행정해석.json",
    ]

    for filepath in processed_files:
        if not filepath.exists():
            print(f"  ⚠️ 파일 없음: {filepath.name}")
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            count = 0
            for item in data:
                text = item.get('text', '')
                if text and len(text) > 30:
                    metadata = item.get('metadata', {})
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': metadata.get('source', filepath.stem),
                            'title': metadata.get('title', ''),
                            'category': metadata.get('category', ''),
                            'url': metadata.get('url', '')
                        }
                    )
                    documents.append(doc)
                    count += 1
            print(f"  ✅ {filepath.name} → {count}개 문서")
        except Exception as e:
            print(f"  ⚠️ 로드 오류 ({filepath.name}): {e}")

    # ==========================================================
    # 2. raw 폴더: 법령 데이터 (새 구조)
    #    {meta_info, body: [{article_text_full, ...}], addenda, tables}
    # ==========================================================
    law_files = [
        data_path / "raw" / "rd_노동법.json",
        data_path / "raw" / "rd_민사법.json",
        data_path / "raw" / "rd_형사법.json",
    ]

    for filepath in law_files:
        if not filepath.exists():
            print(f"  ⚠️ 파일 없음: {filepath.name}")
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            count = 0
            for law in data:
                # meta_info에서 법령 정보 추출
                meta_info = law.get('meta_info', {})
                law_name = meta_info.get('law_name', '')
                category = meta_info.get('category', '')

                # body에서 조문 추출
                body = law.get('body', [])
                for article in body:
                    article_text = article.get('article_text_full', '')
                    article_title = article.get('article_title', '')

                    if article_text and len(article_text) > 30:
                        text = f"[{law_name}] {article_title}\n\n{article_text}"
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': 'law',
                                'law_name': law_name,
                                'law_id': meta_info.get('law_id', ''),
                                'article_no': article.get('article_no', ''),
                                'article_title': article_title,
                                'category': category,
                                'enforce_date': meta_info.get('enforce_date', ''),
                                'revision_type': meta_info.get('revision_type', '')
                            }
                        )
                        documents.append(doc)
                        count += 1
            print(f"  ✅ {filepath.name} → {count}개 조문")
        except Exception as e:
            print(f"  ⚠️ 법령 로드 오류 ({filepath.name}): {e}")

    # ==========================================================
    # 3. PDF 파일
    # ==========================================================
    try:
        pdf_loader = DirectoryLoader(
            path=str(data_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False,
            use_multithreading=True,
            silent_errors=True
        )
        pdf_docs = pdf_loader.load()
        if pdf_docs:
            print(f"  ✅ PDF: {len(pdf_docs)}개 페이지")
        documents.extend(pdf_docs)
    except Exception as e:
        pass

    print(f"\n📄 총 {len(documents)}개 문서 로드 완료\n")
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
    # ---------------------------------------------------------
    # Ragas가 내부적으로 temperature를 0.01 등으로 낮추려 해도
    # 이 클래스가 가로채서 1로 강제 고정합니다.
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
    print(f"📝 테스트셋 생성 중 (목표: {test_size}개)")
    print("   노동법 특성상 조건부/추론 질문이 자동으로 많이 생성됩니다.")
    print()

    # ---------------------------------------------------------
    # Ragas 0.4.x: generate_with_langchain_docs 메서드 사용
    # ---------------------------------------------------------
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=test_size,
        raise_exceptions=False,
    )

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
        '--data-dir',
        type=str,
        default='../data',
        help='PDF/TXT 문서가 있는 폴더 경로 (기본값: ../data)'
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
        default='gpt-5.2',
        help='사용할 LLM 모델 (기본값: gpt-5.2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='labor_law_golden_set.csv',
        help='출력 파일명 (기본값: labor_law_golden_set.csv)'
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
    documents = load_documents(args.data_dir)

    if not documents:
        print("❌ 로드된 문서가 없습니다. 데이터 폴더를 확인해주세요.")
        return

    # ---------------------------------------------------------
    # [추가] 과도한 비용/시간 방지를 위한 랜덤 샘플링
    # ---------------------------------------------------------
    if args.sample_size > 0 and len(documents) > args.sample_size:
        import random
        print(
            f"✂️  문서가 너무 많아 {args.sample_size}개로 샘플링합니다. (전체: {len(documents)}개)")
        documents = random.sample(documents, args.sample_size)

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
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

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
