"""
고용노동부 FAQ 크롤러
URL: https://www.moel.go.kr/faq/faqList.do
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime
import os

# 세션
session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

BASE_URL = "https://www.moel.go.kr"
LIST_URL = "https://www.moel.go.kr/faq/faqList.do"


def get_last_page_number():
    """마지막 페이지 번호 찾기"""
    try:
        response = session.get(LIST_URL, headers=headers)
        response.encoding = 'utf-8'

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # div.paging 내의 마지막 페이지 링크 찾기
            paging_div = soup.select_one('div.paging')
            if paging_div:
                # a.page_last에서 페이지 번호 추출
                last_link = paging_div.select_one('a.page_last')
                if last_link:
                    href = last_link.get('href', '')
                    match = re.search(r'pageIndex=(\d+)', href)
                    if match:
                        return int(match.group(1))

                    # href에서 못찾으면 텍스트에서 숫자 추출
                    text = last_link.get_text()
                    num_match = re.search(r'(\d+)', text)
                    if num_match:
                        return int(num_match.group(1))

            # 대체: 모든 페이지 링크에서 최대값 찾기
            page_links = soup.select('div.paging a[href*="pageIndex"]')
            if page_links:
                last_page = 1
                for link in page_links:
                    href = link.get('href', '')
                    match = re.search(r'pageIndex=(\d+)', href)
                    if match:
                        last_page = max(last_page, int(match.group(1)))
                return last_page

    except Exception as e:
        print(f"마지막 페이지 조회 에러: {e}")

    return 22  # 기본값


def get_faq_list_page(page=1):
    """FAQ 목록 페이지 가져오기"""
    params = {"pageIndex": page}

    try:
        response = session.get(LIST_URL, headers=headers, params=params)
        response.encoding = 'utf-8'

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # 테이블 tbody 내의 tr들
            rows = soup.select('.tstyle_list tbody tr')

            faqs = []
            for row in rows:
                # 링크 추출 (a.ellipsis 또는 strong.b_tit 내의 a)
                link = row.select_one(
                    'a.ellipsis') or row.select_one('strong.b_tit a')
                if link:
                    href = link.get('href', '')
                    title = link.get('title', '') or link.get_text(strip=True)

                    # seqRepeat 추출
                    seq_match = re.search(r'seqRepeat=(\d+)', href)
                    seq = seq_match.group(1) if seq_match else None

                    if not seq:
                        # onclick에서 추출 시도
                        onclick = link.get('onclick', '')
                        seq_match = re.search(r"fnView\('(\d+)'\)", onclick)
                        seq = seq_match.group(1) if seq_match else None

                    if seq:
                        # 카테고리 추출
                        cols = row.select('td')
                        category = ''
                        if len(cols) >= 2:
                            category = cols[1].get(
                                'title', '') or cols[1].get_text(strip=True)

                        faqs.append({
                            'seq': seq,
                            'title': title,
                            'category': category,
                            'url': f"{BASE_URL}/faq/faqView.do?seqRepeat={seq}",
                        })

            return faqs
    except Exception as e:
        print(f"목록 조회 에러 (페이지 {page}): {e}")

    return []


def get_faq_detail(seq):
    """FAQ 상세 내용 가져오기"""
    url = f"{BASE_URL}/faq/faqView.do?seqRepeat={seq}"

    try:
        response = session.get(url, headers=headers)
        response.encoding = 'utf-8'

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            result = {
                'category': '',
                'question': '',
                'answer': '',
            }

            # div.b_info 내의 dl.w100p 요소들 파싱
            info_div = soup.select_one('div.b_info')
            if info_div:
                dls = info_div.select('dl.w100p')

                for dl in dls:
                    dt = dl.select_one('dt')
                    dd = dl.select_one('dd')

                    if dt and dd:
                        label = dt.get_text(strip=True)
                        # dd의 텍스트 추출 (br 태그를 줄바꿈으로 변환)
                        # br 태그 처리
                        for br in dd.find_all('br'):
                            br.replace_with('\n')
                        content = dd.get_text(strip=False).strip()

                        if label == '카테고리':
                            result['category'] = content
                        elif label == '질의':
                            result['question'] = content
                        elif label == '답변':
                            result['answer'] = content

            return result

    except Exception as e:
        print(f"상세 조회 에러 (seq={seq}): {e}")

    return None


def clean_text(text):
    """텍스트 정리"""
    if not text:
        return ""

    # 연속 줄바꿈 정규화
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 연속 공백 정규화
    text = re.sub(r' {2,}', ' ', text)
    # 줄 앞뒤 공백 제거
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def crawl_all_faqs():
    """전체 FAQ 크롤링"""
    print("=== 고용노동부 FAQ 크롤링 시작 ===\n")

    # 마지막 페이지 확인
    last_page = get_last_page_number()
    print(f"총 {last_page} 페이지 발견\n")

    # Step 1: 목록 수집
    print("[Step 1] FAQ 목록 수집 중...")
    all_faqs = []

    for page in range(1, last_page + 1):
        print(f"  페이지 {page}/{last_page} 조회 중...")
        faqs = get_faq_list_page(page)

        if faqs:
            all_faqs.extend(faqs)
            print(f"  → {len(faqs)}개 FAQ (누적: {len(all_faqs)}개)")
        else:
            print(f"  → 결과 없음")

        time.sleep(0.3)  # 서버 부하 방지

    print(f"\n총 {len(all_faqs)}개 FAQ 발견\n")

    if not all_faqs:
        print("FAQ를 가져오지 못했습니다.")
        return []

    # Step 2: 상세 내용 크롤링
    print("[Step 2] 상세 내용 크롤링 중...")
    results = []

    for i, faq in enumerate(all_faqs):
        print(f"  [{i+1}/{len(all_faqs)}] {faq['title'][:40]}...")

        detail = get_faq_detail(faq['seq'])

        if detail:
            results.append({
                'seq': faq['seq'],
                'title': faq['title'],
                'category': detail['category'] or faq.get('category', ''),
                'question': clean_text(detail['question']),
                'answer': clean_text(detail['answer']),
                'url': faq['url'],
            })
            print(f"    ✓ 카테고리: {detail['category']}")
        else:
            print(f"    ✗ 상세 내용 없음")

        time.sleep(0.3)

    return results


def save_results(data, filename):
    """결과 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장: {filename}")


def main():
    results = crawl_all_faqs()

    if results:
        # 저장 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data', 'processed')
        os.makedirs(data_dir, exist_ok=True)

        # 타임스탬프로 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(data_dir, f'moel_faq_{timestamp}.json')
        save_results(results, filepath)

        # 통계
        print("\n=== 크롤링 완료 ===")
        print(f"총 {len(results)}개 FAQ")

        # 카테고리별 통계
        categories = {}
        for r in results:
            cat = r['category'] or '미분류'
            categories[cat] = categories.get(cat, 0) + 1

        print("\n[카테고리별 현황]")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  • {cat}: {count}개")

        print(f"\n저장 위치: {filepath}")


if __name__ == "__main__":
    main()
