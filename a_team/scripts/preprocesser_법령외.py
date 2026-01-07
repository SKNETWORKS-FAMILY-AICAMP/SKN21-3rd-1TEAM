"""
법령 외 데이터(판례, 행정해석) 전처리 및 Qdrant 업로드 스크립트
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def clean_text(text: str) -> str:
    """공통 텍스트 정리 함수"""
    if not text:
        return ""

    # "목록" 텍스트 제거
    text = re.sub(r'\n*목록$', '', text)

    # 연속 줄바꿈 정규화 (3개 이상 → 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 앞뒤 공백 제거
    return text.strip()


def load_case_law_data(file_path: Path) -> Tuple[List[str], List[Dict]]:
    """
    주요판례 데이터 로드 및 전처리

    Args:
        file_path: rd_주요판례.json 파일 경로

    Returns:
        (documents, metadatas) 튜플
    """
    print(f"\n📂 판례 데이터 로드 중: {file_path.name}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    metadatas = []

    for item in tqdm(data, desc="판례 전처리"):
        제목 = item.get('제목', '').strip()
        판정사항 = clean_text(item.get('판정사항', ''))
        판정요지 = clean_text(item.get('판정요지', ''))

        # 판정사항 또는 판정요지가 있어야 함
        if not (판정사항 or 판정요지):
            continue

        # 텍스트 구성: [판례: 제목]\n판정사항\n판정요지
        text_parts = [f"[판례: {제목}]"]

        if 판정사항:
            text_parts.append(f"판정사항: {판정사항}")

        if 판정요지:
            text_parts.append(f"판정요지: {판정요지}")

        text = "\n\n".join(text_parts)

        # 문서 및 메타데이터 추가
        documents.append(text)
        metadatas.append({
            'source': 'case_law',
            'title': 제목,
            'category': item.get('자료구분', ''),
            'department': item.get('담당부서', ''),
            'reg_date': item.get('등록일', ''),
            'number': str(item.get('번호', '')),
            'doc_length': len(text)
        })

    print(f"✅ 판례 {len(documents)}개 문서 전처리 완료")
    return documents, metadatas


def load_interpretation_data(file_path: Path) -> Tuple[List[str], List[Dict]]:
    """
    행정해석 데이터 로드 및 전처리

    Args:
        file_path: rd_행정해석.json 파일 경로

    Returns:
        (documents, metadatas) 튜플
    """
    print(f"\n📂 행정해석 데이터 로드 중: {file_path.name}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    metadatas = []

    for item in tqdm(data, desc="행정해석 전처리"):
        title = item.get('title', '').strip()

        # parsed 필드 사용 (깨끗하게 정제된 질의-답변)
        parsed = item.get('parsed', {})

        # parse 실패 시 스킵
        if not parsed.get('parse_success', False):
            continue

        questions = parsed.get('questions', [])
        answers = parsed.get('answers', [])

        # 질의와 답변이 모두 있어야 함
        if not (questions and answers):
            continue

        # 첫 번째 질의-답변 사용
        question = clean_text(questions[0])
        answer = clean_text(answers[0])

        # 텍스트 구성: [행정해석] 제목\n질의\n답변
        text = f"[행정해석] {title}\n\n질의:\n{question}\n\n답변:\n{answer}"

        # 문서 및 메타데이터 추가
        documents.append(text)
        metadatas.append({
            'source': 'interpretation',
            'title': title,
            'department': item.get('department', ''),
            'person': item.get('person', ''),
            'reg_date': item.get('reg_date', ''),
            'number': item.get('number', ''),
            'url': item.get('url', ''),
            'doc_length': len(text)
        })

    print(f"✅ 행정해석 {len(documents)}개 문서 전처리 완료")
    return documents, metadatas


def load_moel_qa_data(file_path: Path) -> Tuple[List[str], List[Dict]]:
    """
    고용노동부 Q&A 데이터 로드 및 전처리

    Args:
        file_path: rd_법령외_고용노동부QA.json 파일 경로

    Returns:
        (documents, metadatas) 튜플
    """
    print(f"\n📂 고용노동부 Q&A 데이터 로드 중: {file_path.name}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    metadatas = []

    for item in tqdm(data, desc="Q&A 전처리"):
        title = item.get('title', '').strip()
        question = clean_text(item.get('question', ''))
        answer = clean_text(item.get('answer', ''))

        # 질의와 답변이 모두 있어야 함
        if not (question and answer):
            continue

        # 텍스트 구성: [고용노동부 Q&A] 제목\n질의\n답변
        text = f"[고용노동부 Q&A] {title}\n\n질의:\n{question}\n\n답변:\n{answer}"

        # 문서 및 메타데이터 추가
        documents.append(text)
        metadatas.append({
            'source': 'moel_qa',
            'title': title,
            'category': item.get('category', ''),
            'seq': item.get('seq', ''),
            'url': item.get('url', ''),
            'doc_length': len(text)
        })

    print(f"✅ 고용노동부 Q&A {len(documents)}개 문서 전처리 완료")
    return documents, metadatas


def load_qa_response_data(file_path: Path) -> Tuple[List[str], List[Dict]]:
    """
    중앙부처 1차 해석 (질의회답) 데이터 로드 및 전처리

    Args:
        file_path: rd_법령외_질의회답.json 파일 경로

    Returns:
        (documents, metadatas) 튜플
    """
    print(f"\n📂 중앙부처 1차 해석 데이터 로드 중: {file_path.name}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    metadatas = []

    for item in tqdm(data, desc="질의회답 전처리"):
        title = item.get('title', '').strip()
        question = clean_text(item.get('question', ''))
        answer = clean_text(item.get('answer', ''))

        # 질의와 답변은 필수
        if not (question or answer):
            continue

        # 텍스트 구성
        text = f"[질의회답] {title}\n\n질의:\n{question}\n\n답변:\n{answer}"

        # 문서 및 메타데이터 추가
        documents.append(text)
        metadatas.append({
            'source': 'qa_response',
            'title': title,
            'agency': item.get('agency', ''),
            'date': item.get('date', ''),
            'url': item.get('url', ''),
            'doc_length': len(text)
        })

    print(f"✅ 중앙부처 1차 해석 {len(documents)}개 문서 전처리 완료")
    return documents, metadatas


def save_preprocessed_data(documents: List[str], metadatas: List[Dict], output_path: Path):
    """
    전처리된 데이터를 JSON 파일로 저장

    Args:
        documents: 전처리된 문서 리스트
        metadatas: 메타데이터 리스트
        output_path: 저장할 파일 경로
    """
    print(f"\n💾 전처리 데이터 저장 중: {output_path}")

    output_data = []
    for doc, meta in zip(documents, metadatas):
        output_data.append({
            'text': doc,
            'metadata': meta
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장 완료: {len(output_data)}개 문서")


def main():
    """메인 실행 함수"""

    print("\n" + "="*60)
    print("📝 판례·행정해석 데이터 전처리")
    print("="*60)

    # 경로 설정
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # processed 디렉토리 생성
    processed_dir.mkdir(exist_ok=True, parents=True)

    # 입력 파일
    case_law_file = raw_dir / "rd_법령외_주요판례.json"
    interpretation_file = raw_dir / "rd_법령외_행정해석.json"
    moel_qa_file = raw_dir / "rd_법령외_고용노동부QA.json"

    # 출력 파일
    case_law_output = processed_dir / "fd_법령외_판례.json"
    interpretation_output = processed_dir / "fd_법령외_행정해석.json"
    moel_qa_output = processed_dir / "fd_법령외_고용노동부QA.json"

    # 파일 존재 확인
    if not case_law_file.exists():
        print(f"❌ 파일 없음: {case_law_file}")
        return

    if not interpretation_file.exists():
        print(f"❌ 파일 없음: {interpretation_file}")
        return

    if not moel_qa_file.exists():
        print(f"❌ 파일 없음: {moel_qa_file}")
        return

    # 데이터 로드 및 전처리
    # 1. 판례 데이터
    case_docs, case_metas = load_case_law_data(case_law_file)

    # 2. 행정해석 데이터
    interp_docs, interp_metas = load_interpretation_data(interpretation_file)

    # 3. 고용노동부 Q&A 데이터
    moel_qa_docs, moel_qa_metas = load_moel_qa_data(moel_qa_file)

    # 4. 중앙부처 1차 해석 (판정선례) 데이터
    qa_response_file = raw_dir / "rd_법령외_판정선례.json"
    qa_response_output = processed_dir / "fd_법령외_판정선례.json"

    if qa_response_file.exists():
        qa_resp_docs, qa_resp_metas = load_qa_response_data(qa_response_file)
        save_preprocessed_data(qa_resp_docs, qa_resp_metas, qa_response_output)
        print(f"  - {qa_response_output}")
    else:
        print(f"⚠️ 파일 없음: {qa_response_file} (건너뜀)")

    # 전처리 데이터 저장 (판례)
    save_preprocessed_data(case_docs, case_metas, case_law_output)

    # 전처리 데이터 저장 (행정해석)
    save_preprocessed_data(interp_docs, interp_metas, interpretation_output)

    # 전처리 데이터 저장 (고용노동부 Q&A)
    save_preprocessed_data(moel_qa_docs, moel_qa_metas, moel_qa_output)

    print("\n" + "="*60)
    print("✅ 전처리 완료!")
    print(f"📄 출력 파일:")
    print(f"  - {case_law_output}")
    print(f"  - {interpretation_output}")
    print(f"  - {moel_qa_output}")
    print("="*60 + "\n")

    # 샘플 출력
    print("📄 샘플 문서 (판례):")
    print("-"*60)
    print(case_docs[0][:500] + "..." if len(case_docs[0])
          > 500 else case_docs[0])
    print("-"*60)

    print("\n📄 샘플 문서 (행정해석):")
    print("-"*60)
    print(interp_docs[0][:500] + "..." if len(interp_docs[0])
          > 500 else interp_docs[0])
    print("-"*60)

    print("\n📄 샘플 문서 (고용노동부 Q&A):")
    print("-"*60)
    print(moel_qa_docs[0][:500] +
          "..." if len(moel_qa_docs[0]) > 500 else moel_qa_docs[0])
    print("-"*60)


if __name__ == "__main__":
    main()
