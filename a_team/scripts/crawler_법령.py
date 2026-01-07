import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import json
import time
import re
import os

# 데이터 저장 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# 디렉토리 생성
os.makedirs(RAW_DIR, exist_ok=True)

# Session for list API
session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://www.law.go.kr",
    "Referer": "https://www.law.go.kr/lsAstSc.do?menuId=391&subMenuId=397&tabMenuId=437&query=",
}

AJAX_LIST_URL = "https://www.law.go.kr/lsAstScListR.do"

# 세 가지 분야 정의 (이미지에서 확인한 값들)
CATEGORIES = [
    {
        "name": "노동법",
        "lsFdCd": "40,40010000,40020000,40030000,40040000,40050000,40060000,40070000",
        "p5": "40,40010000,40020000,40030000,40040000,40050000,40060000,40070000",
    },
    {
        "name": "민사법",
        "lsFdCd": "08,08010000,08020000,08030000,08030100,08030200,08030300,08030400,08040000,08050000,08060000",
        "p5": "08,08010000,08020000,08030000,08030100,08030200,08030300,08030400,08040000,08050000,08060000",
    },
    {
        "name": "형사법",
        "lsFdCd": "09,09010000,09020000,09030000,09040000,09040100,09040200,09040300,09040400,09050000,09060000",
        "p5": "09,09010000,09020000,09030000,09040000,09040100,09040200,09040300,09040400,09050000,09060000",
    },
]


def get_list_params(page=1, category=None):
    """AJAX 요청 파라미터 - 분야별로 다른 파라미터 사용"""
    if category is None:
        category = CATEGORIES[0]

    return {
        "lsFdCd": category["lsFdCd"],
        "pg": str(page),
        "nwHst": "",
        "q": "*",
        "outmax": "50",
        "p9": "2,4",
        "p5": category["p5"],
        "p18": "0",
        "lsFdSave": "N",
        "cptOfiMgDptChk": "N",
        "p19": "1,3",
        "menuId": "391",
        "subMenuId": "397",
        "tabMenuId": "437",
    }


def get_law_list_page(page=1, category=None):
    """AJAX로 특정 페이지의 법령 목록 가져오기"""
    params = get_list_params(page, category)

    try:
        response = session.post(AJAX_LIST_URL, headers=headers, data=params)
        response.encoding = 'utf-8'

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.select('#resultTable > tbody > tr')

            laws = []
            for row in rows:
                link = row.select_one('a[onclick]')
                if link:
                    title = link.get_text(strip=True)
                    onclick = link.get('onclick', '')

                    match = re.findall(r"'(.*?)'", onclick)
                    if len(match) >= 4:
                        laws.append({
                            'title': title,
                            'lsi_seq': match[3],
                            'ef_yd': match[1],
                            'category': category["name"] if category else "unknown",
                        })

            return laws
    except Exception as e:
        print(f"목록 조회 에러 (페이지 {page}): {e}")

    return []


def get_all_law_list_for_category(category):
    """특정 분야의 모든 페이지를 순회하며 법령 목록 가져오기"""
    all_laws = []
    page = 1

    while True:
        print(f"    페이지 {page} 조회 중...")
        laws = get_law_list_page(page, category)

        if not laws:
            print(f"    → 페이지 {page}에서 결과 없음. 수집 종료.")
            break

        all_laws.extend(laws)
        print(f"    → {len(laws)}개 법령 수집 (누적: {len(all_laws)}개)")

        page += 1
        time.sleep(0.3)  # 서버 부하 방지

    return all_laws


def get_all_law_list():
    """모든 분야의 모든 페이지를 순회하며 전체 법령 목록 가져오기"""
    all_laws = []

    for i, category in enumerate(CATEGORIES):
        print(f"\n  [{i+1}/{len(CATEGORIES)}] {category['name']} 분야 크롤링...")
        laws = get_all_law_list_for_category(category)
        all_laws.extend(laws)
        print(f"  → {category['name']} 완료: {len(laws)}개 법령")

    return all_laws


def get_law_details_with_playwright(laws, max_count=5):
    """Playwright로 상세 페이지 크롤링 (JS 렌더링 필요)"""

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, law in enumerate(laws[:max_count]):
            print(
                f"[{i+1}/{max_count}] [{law.get('category', '')}] {law['title'][:35]}... 크롤링 중")

            try:
                detail_url = f"https://www.law.go.kr/lsInfoP.do?lsiSeq={law['lsi_seq']}&efYd={law['ef_yd']}&ancYnChk=0&nwJoYnInfo=Y&efGubun=Y&vSct=*"

                page.goto(detail_url)
                page.wait_for_selector('#conScroll', timeout=15000)
                time.sleep(1)

                law_data = {
                    'title': law['title'],
                    'lsi_seq': law['lsi_seq'],
                    'category': law.get('category', 'unknown'),
                    'url': detail_url,
                    'articles': []
                }

                # 본조항 및 부칙 크롤링
                main_articles = page.evaluate('''() => {
                    const results = [];
                    const pgroups = document.querySelectorAll('#conScroll .pgroup');
                    pgroups.forEach((el, idx) => {
                        const isAddendum = el.closest('#arDivArea') !== null;
                        results.push({
                            text: el.innerText.trim(),
                            isAddendum: isAddendum
                        });
                    });
                    return results;
                }''')

                article_count = 0
                addendum_count = 0

                for j, item in enumerate(main_articles):
                    try:
                        text = item['text']
                        is_addendum = item['isAddendum']

                        if text and len(text) > 10:
                            if is_addendum:
                                addendum_count += 1
                                article_num = f"부칙 {addendum_count}"
                            else:
                                article_match = re.search(
                                    r'제\d+조(?:의\d+)?', text)
                                article_num = article_match.group(
                                ) if article_match else f"항목{j+1}"
                                article_count += 1

                            law_data['articles'].append({
                                'article_num': article_num,
                                'is_addendum': is_addendum,
                                'index': j + 1,
                                'content': text
                            })
                    except:
                        continue

                results.append(law_data)
                print(f"  → 본조항 {article_count}개, 부칙 {addendum_count}개 수집 완료")

                time.sleep(0.5)

            except Exception as e:
                print(f"  에러 발생: {e}")
                continue

        browser.close()

    return results


def save_results(data, filename='law_data.json'):
    """결과 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n결과가 {filename}에 저장되었습니다.")


def main():
    print("=== 법령 크롤링 시작 ===")
    print(f"크롤링 대상: {len(CATEGORIES)}개 분야\n")

    # Step 1: 모든 분야의 법령 목록 가져오기
    print("[Step 1] 모든 분야 법령 목록 조회 중...")
    laws = get_all_law_list()
    print(f"\n총 {len(laws)}개 법령 발견\n")

    if not laws:
        print("목록을 가져오지 못했습니다.")
        return

    # Step 2: Playwright로 상세 페이지 크롤링
    print("[Step 2] 상세 페이지 크롤링 (Playwright)...")
    results = get_law_details_with_playwright(laws, max_count=len(laws))

    # Step 3: 결과 저장 - raw 폴더
    if results:
        # 전체 통합 파일 저장
        save_results(results, os.path.join(RAW_DIR, 'law_data_all.json'))

        # 분야별 별도 파일 저장
        for cat in CATEGORIES:
            cat_results = [r for r in results if r['category'] == cat['name']]
            if cat_results:
                filename = os.path.join(RAW_DIR, f"{cat['name']}.json")
                save_results(cat_results, filename)

        print(f"\n=== 크롤링 완료 ===")
        print(
            f"총 {len(results)}개 법령, {sum(len(r['articles']) for r in results)}개 조항 수집")

        # 분야별 통계
        for cat in CATEGORIES:
            cat_count = len(
                [r for r in results if r['category'] == cat['name']])
            cat_articles = sum(len(r['articles'])
                               for r in results if r['category'] == cat['name'])
            print(
                f"  - {cat['name']}: {cat_count}개 법령, {cat_articles}개 조항 → {cat['name']}.json")
    else:
        print("크롤링된 데이터가 없습니다.")


if __name__ == "__main__":
    main()
