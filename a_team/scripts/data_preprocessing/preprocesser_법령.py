"""
법령 데이터 전처리 스크립트
- Raw JSON 데이터 로드 (노동법/민사법/형사법)
- 텍스트 정제 및 청킹
- 부칙 처리 (최신 부칙만 유지)
- 별표 처리 (관련 조문 병합 또는 독립 청크)
- 결과 저장: processed/law_chunks.json
"""

import json
import os
import re
from typing import List, Dict, Any
from pathlib import Path

# ============================================================
# 설정
# ============================================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / '..' / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# 청킹 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TABLE_MERGE_THRESHOLD = 300  # 별표 병합 기준 (글자 수)


# ============================================================
# 유틸리티 함수
# ============================================================
def load_json(filepath: Path) -> Any:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: Path):
    """JSON 파일 저장"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 저장 완료: {filepath}")


def clean_text(text: str) -> str:
    """텍스트 정제"""
    if not text:
        return ""

    # HTML 태그 제거 (간단한 정규식, 필요시 라이브러리 사용)
    text = re.sub(r'<[^>]+>', '', text)

    # 개정 이력 태그 간소화: <개정 2021. 1. 5.> -> [개정 2021.1.5]
    text = re.sub(r'<개정\s*([^>]+)>', r'[개정 \1]', text)

    # 연속 공백/줄바꿈 정규화
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # 헤더 정규화 (예: 부    칙 -> 부칙, 별    표 -> 별표)
    text = re.sub(r'부\s+칙', '부칙', text)
    text = re.sub(r'별\s+표', '별표', text)

    return text.strip()


def split_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """텍스트 오버랩 분할"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # 문장/단어 경계 처리
        if end < len(text):
            for sep in ['. ', ', ', ' ', '\n']:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start + chunk_size // 2:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text) - overlap:
            break

    return chunks


# ============================================================
# 청킹 로직
# ============================================================
def process_law_data(law: Dict[str, Any]) -> List[Dict[str, Any]]:
    """단일 법령 데이터 처리 (조문, 부칙, 별표)"""
    processed_chunks = []
    law_meta = law.get('meta_info', {})

    # 1. 메타데이터 추출
    base_meta = {
        'source': 'law',
        'law_name': law_meta.get('law_name', ''),
        'law_id': law_meta.get('law_id', ''),
        'category': law_meta.get('category', ''),
        'enforce_date': law_meta.get('enforce_date', ''),
        'revision_type': law_meta.get('revision_type', ''),
        'url': law_meta.get('url', '')
    }

    law_name = base_meta['law_name']

    # ------------------------------------------------------------
    # 2. 별표 (Table) 전처리: 조문 병합용 매핑 생성
    # ------------------------------------------------------------
    table_map = {}  # { article_no: [table_content, ...] }
    independent_tables = []

    for table in law.get('tables', []):
        raw_html = table.get('content_html', '')
        # HTML에서 텍스트만 대략 추출 (다운로드 링크 등 제외)
        # 실제로는 제목이 가장 중요함: [별표 1] ...
        # 간단하게 태그 제거 후 정제
        table_text = clean_text(raw_html)

        # 제목 추출 시도 (raw 데이터 구조에 따라 다를 수 있음, 여기선 HTML title 속성이나 텍스트 앞부분 활용)
        # 별표 텍스트가 [별표 1] ... 제50조 관련 ... 형식을 띤다고 가정

        # 관련 조문 찾기 (예: "제50조", "제50조의2")
        # 정규식: 제([0-9]+(의[0-9]+)?)조
        match = re.search(r'제(\d+(?:의\d+)?)조', table_text)
        related_article_no = match.group(1) if match else None

        # 조건: 관련 조문이 있고 + 길이가 짧으면 => 병합 대상
        if related_article_no and len(table_text) < TABLE_MERGE_THRESHOLD:
            if related_article_no not in table_map:
                table_map[related_article_no] = []
            table_map[related_article_no].append(table_text)
        else:
            # 독립 청크로 처리
            independent_tables.append(table_text)

    # ------------------------------------------------------------
    # 3. 조문 (Body) 처리 - 계층적 청킹
    # ------------------------------------------------------------
    for article in law.get('body', []):
        article_no = article.get('article_no', '')
        article_title = article.get('article_title', '')
        paragraphs = article.get('paragraphs', [])

        if not paragraphs:
            continue

        # 별표 내용 병합
        table_text = ""
        if article_no in table_map:
            merged_tables = "\n\n".join(table_map[article_no])
            table_text = f"\n\n[관련 별표]\n{merged_tables}"

        # 헤더 생성
        context_header = f"[{law_name}] {article_title}\n\n"

        # 메타데이터 구성
        chunk_meta = base_meta.copy()
        chunk_meta.update({
            'article_no': article_no,
            'article_title': article_title,
            'type': 'article'
        })

        # ========== Level 1: 조문 전체를 1개 청크로 시도 ==========
        full_article_text = "\n\n".join(
            [p.get('content', '') for p in paragraphs]) + table_text
        full_text = context_header + clean_text(full_article_text)

        if len(full_text) <= CHUNK_SIZE:
            # Level 1 성공: 조문 전체가 1개 청크에 들어감
            processed_chunks.append({
                'text': full_text,
                'metadata': {
                    **chunk_meta,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'is_continuation': False,
                    'chunking_level': 'article'  # 전체 조문
                }
            })
        else:
            # ========== Level 2: 항(paragraph) 단위로 분할 시도 ==========
            paragraph_chunks = []
            current_chunk = ""
            current_paragraphs = []

            for para in paragraphs:
                para_content = clean_text(para.get('content', ''))
                if not para_content:
                    continue

                # 현재 청크에 이 항을 추가했을 때의 길이 계산
                test_chunk = current_chunk + "\n\n" + \
                    para_content if current_chunk else para_content
                test_full = context_header + test_chunk + table_text

                if len(test_full) <= CHUNK_SIZE:
                    # 추가 가능
                    current_chunk = test_chunk
                    current_paragraphs.append(para)
                else:
                    # 현재 청크를 저장하고 새로운 청크 시작
                    if current_chunk:
                        paragraph_chunks.append({
                            'text': current_chunk,
                            'paragraphs': current_paragraphs.copy()
                        })

                    # 이 항으로 새로운 청크 시작
                    current_chunk = para_content
                    current_paragraphs = [para]

                    # 만약 단일 항도 너무 길면 Level 3으로 이동
                    single_para_full = context_header + para_content + table_text
                    if len(single_para_full) > CHUNK_SIZE:
                        # ========== Level 3: 호(subparagraph) 단위 분할 필요 ==========
                        # 일단 항 자체를 추가하고, 나중에 처리
                        pass

            # 마지막 청크 추가
            if current_chunk:
                paragraph_chunks.append({
                    'text': current_chunk,
                    'paragraphs': current_paragraphs.copy()
                })

            # Level 2 청크들을 processed_chunks에 추가
            total = len(paragraph_chunks)
            for idx, pchunk in enumerate(paragraph_chunks):
                is_cont = (idx > 0)

                # 별표는 마지막 청크에만 추가
                chunk_table = table_text if idx == total - 1 else ""
                text_content = context_header + pchunk['text'] + chunk_table

                if is_cont:
                    text_content = f"{context_header}[이어짐 {idx+1}/{total}]\n{pchunk['text']}{chunk_table}"

                processed_chunks.append({
                    'text': text_content,
                    'metadata': {
                        **chunk_meta,
                        'chunk_index': idx,
                        'total_chunks': total,
                        'is_continuation': is_cont,
                        'chunking_level': 'paragraph'  # 항 단위
                    }
                })

    # ------------------------------------------------------------
    # 4. 독립 별표 (Independent Tables) 처리
    # ------------------------------------------------------------
    for i, table_text in enumerate(independent_tables):
        # 긴 별표는 자체적으로 청킹
        header = f"[{law_name}] 별표/서식 {i+1}\n\n"
        full_text = header + table_text

        chunk_meta = base_meta.copy()
        chunk_meta.update({
            'article_no': f"별표{i+1}",
            'article_title': "별표/서식",
            'type': 'table'
        })

        if len(full_text) <= CHUNK_SIZE:
            processed_chunks.append({
                'text': full_text,
                'metadata': {**chunk_meta, 'chunk_index': 0, 'total_chunks': 1}
            })
        else:
            splits = split_with_overlap(
                table_text, CHUNK_SIZE - len(header) - 30, CHUNK_OVERLAP)
            for j, split in enumerate(splits):
                text_content = f"{header}[이어짐 {j+1}/{len(splits)}]\n{split}"
                processed_chunks.append({
                    'text': text_content,
                    'metadata': {
                        **chunk_meta,
                        'chunk_index': j,
                        'total_chunks': len(splits),
                        'is_continuation': j > 0
                    }
                })

    # ------------------------------------------------------------
    # 5. 부칙 (Addenda) 처리 - 최신 1개만
    # ------------------------------------------------------------
    addenda = law.get('addenda', [])
    if addenda:
        last_addendum = addenda[-1]
        t = clean_text(last_addendum.get('content', ''))
        title = clean_text(last_addendum.get('article_title', '부칙'))

        # 헤더
        header = f"[{law_name}] {title}\n\n"
        full_text = header + t

        chunk_meta = base_meta.copy()
        chunk_meta.update({
            'article_no': '부칙',
            'article_title': title,
            'type': 'addendum'
        })

        if len(full_text) <= CHUNK_SIZE:
            processed_chunks.append({
                'text': full_text,
                'metadata': {**chunk_meta, 'chunk_index': 0, 'total_chunks': 1}
            })
        else:
            splits = split_with_overlap(
                t, CHUNK_SIZE - len(header) - 30, CHUNK_OVERLAP)
            for j, split in enumerate(splits):
                text_content = f"{header}[이어짐 {j+1}/{len(splits)}]\n{split}"
                processed_chunks.append({
                    'text': text_content,
                    'metadata': {
                        **chunk_meta,
                        'chunk_index': j,
                        'total_chunks': len(splits),
                        'is_continuation': j > 0
                    }
                })

    return processed_chunks


# ============================================================
# 메인 실행
# ============================================================
def main():
    print("🚀 법령 데이터 전처리 시작...")

    all_chunks = []

    # Raw 디렉토리의 모든 law JSON 파일 찾기
    law_files = list(RAW_DIR.glob('rd_법령_*.json')) + \
        list(RAW_DIR.glob('rd_*.json'))
    # 중복 제거 및 법령외 파일 제외
    law_files = [f for f in set(law_files) if '법령외' not in f.name]

    if not law_files:
        print("⚠️ Raw 디렉토리에 법령 파일이 없습니다.")
        return

    for filepath in sorted(law_files):
        print(f"📂 처리 중: {filepath.name}")
        data = load_json(filepath)

        count = 0
        for law in data:
            chunks = process_law_data(law)
            all_chunks.extend(chunks)
            count += 1

        print(f"   - 법령 수: {count}, 생성된 청크: {len(chunks)} (마지막 법령 기준)")

    print(f"\n📊 총 청크 수: {len(all_chunks)}")

    # 결과 저장
    out_path = PROCESSED_DIR / 'fd_법령_chunked.json'
    save_json(all_chunks, out_path)


if __name__ == '__main__':
    main()
