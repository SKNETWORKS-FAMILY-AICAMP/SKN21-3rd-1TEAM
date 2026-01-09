import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import json
import time
import re
import os

# 데이터 저장 경로 설정
# scripts/crawlers -> scripts -> a_team -> a_team/data/raw
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/crawlers
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')  # a_team/data
RAW_DIR = os.path.join(DATA_DIR, 'raw')  # a_team/data/raw

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
CATEGORY_SEARCH_URL = "https://www.law.go.kr/lsAstSc.do?menuId=391&subMenuId=397&tabMenuId=437&query="

# 기본 카테고리 (자동 추출 실패 시 사용)
DEFAULT_CATEGORIES = [
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


def extract_all_categories():
    """웹사이트에서 모든 법분야 카테고리를 자동으로 추출

    Returns:
        List of category dictionaries with 'name', 'lsFdCd', 'p5' keys
    """
    print("\n[카테고리 자동 추출] 웹사이트에서 모든 법분야 카테고리 추출 중...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            # 법령 검색 페이지 접속
            page.goto(CATEGORY_SEARCH_URL)
            page.wait_for_load_state('networkidle')
            time.sleep(0.5)

            # "법분야별" 탭 클릭 - 정확한 selector 사용 (div.tab3_1 > a:nth-of-type(2))
            law_field_tab = page.locator('div.tab3_1 > a:nth-of-type(2)').first
            law_field_tab.click()
            time.sleep(1)

            # 카테고리 목록이 보일 때까지 대기
            page.wait_for_selector('#divLsFd', state='visible', timeout=10000)

            # 모든 메인 카테고리 추출 (ID가 6자리인 것만: lsFd01, lsFd02, ...)
            categories_data = page.evaluate('''() => {
                const sidebar = document.querySelector('#divLsFd');
                if (!sidebar) return [];
                
                const links = Array.from(sidebar.querySelectorAll('a[id^="lsFd"]'));
                const mainCategories = links.filter(l => l.id.length === 6);
                
                return mainCategories.map(link => {
                    const onclick = link.getAttribute('onclick') || '';
                    const match = onclick.match(/clickLsFd\\('([^']+)'\\)/);
                    const lsFdCd = match ? match[1] : '';
                    
                    return {
                        id: link.id,
                        name: (link.textContent || link.getAttribute('title') || '').trim(),
                        lsFdCd: lsFdCd
                    };
                }).filter(cat => cat.lsFdCd !== '');  // lsFdCd가 있는 것만
            }''')

            browser.close()

            if not categories_data:
                print("  → 카테고리 추출 실패, 기본 카테고리 사용")
                return DEFAULT_CATEGORIES

            # 데이터 포맷팅
            categories = []
            for cat_data in categories_data:
                categories.append({
                    'name': cat_data['name'],
                    'lsFdCd': cat_data['lsFdCd'],
                    'p5': cat_data['lsFdCd'],  # p5도 동일한 값 사용
                })

            print(f"  → {len(categories)}개 카테고리 추출 완료")
            return categories

        except Exception as e:
            print(f"  → 카테고리 추출 중 에러: {e}")
            print("  → 기본 카테고리 사용")
            browser.close()
            return DEFAULT_CATEGORIES


def get_list_params(page=1, category=None):
    """AJAX 요청 파라미터 - 분야별로 다른 파라미터 사용"""
    if category is None:
        category = DEFAULT_CATEGORIES[0]

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


def parse_meta_info(meta_text, law):
    """ct_sub 클래스에서 메타정보 파싱

    예시 입력: "[시행 2026. 1. 2.] [대통령령 제35947호, 2025. 12. 30., 타법개정]"
    """
    meta_info = {
        'law_id': law['lsi_seq'],
        'law_name': law['title'],
        'category_main': law.get('category', 'unknown'),
        'category_main_code': law.get('category_main_code'),
        'category_sub_codes': law.get('category_sub_codes', []),
        'url': f"https://www.law.go.kr/lsInfoP.do?lsiSeq={law['lsi_seq']}&efYd={law.get('ef_yd', '')}",
        'enforce_date': None,
        'promulgation_date': None,
        'promulgation_no': None,
        'revision_type': None
    }

    # 시행일 파싱: [시행 2026. 1. 2.]
    enforce_match = re.search(r'\[시행\s*([\d.\s]+)\]', meta_text)
    if enforce_match:
        date_str = enforce_match.group(1).strip()
        # 2026. 1. 2. → 2026-01-02 형식으로 변환
        date_parts = [p.strip() for p in date_str.split('.') if p.strip()]
        if len(date_parts) >= 3:
            try:
                meta_info['enforce_date'] = f"{date_parts[0]}-{date_parts[1].zfill(2)}-{date_parts[2].zfill(2)}"
            except:
                meta_info['enforce_date'] = date_str

    # 공포정보 파싱: [대통령령 제35947호, 2025. 12. 30., 타법개정]
    # 또는 [법률 제21243호, 2025. 12. 30., 일부개정]
    prom_match = re.search(
        r'\[([^,\]]+제[\d]+호),\s*([\d.\s]+),\s*([^\]]+)\]', meta_text)
    if prom_match:
        meta_info['promulgation_no'] = prom_match.group(1).strip()
        date_str = prom_match.group(2).strip()
        date_parts = [p.strip() for p in date_str.split('.') if p.strip()]
        if len(date_parts) >= 3:
            try:
                meta_info[
                    'promulgation_date'] = f"{date_parts[0]}-{date_parts[1].zfill(2)}-{date_parts[2].zfill(2)}"
            except:
                meta_info['promulgation_date'] = date_str
        meta_info['revision_type'] = prom_match.group(3).strip()

    return meta_info


def parse_paragraphs(content):
    """조문 내용에서 항(①, ②, ③...)을 분리

    원문자 마커로 항을 구분하여 리스트로 반환
    """
    # 항 번호 패턴: 원문자 ① ② ③ ... ⑳
    paragraph_markers = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
    paragraph_pattern = f'([{paragraph_markers}])'

    parts = re.split(paragraph_pattern, content)
    paragraphs = []

    if len(parts) <= 1:
        # 항 구분이 없는 경우 전체를 하나의 항으로
        paragraphs.append({'no': '1', 'content': content.strip()})
    else:
        # 첫 부분 (조문 제목 등)이 있으면 무시하거나 포함
        idx = 0
        para_num = 1

        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                marker = parts[i]
                text = parts[i + 1].strip()
                # 마커와 내용 결합
                full_content = f"{marker} {text}" if text else marker
                paragraphs.append({
                    'no': str(para_num),
                    'content': full_content
                })
                para_num += 1

        # 항이 분리되지 않았으면 전체를 하나로
        if not paragraphs:
            paragraphs.append({'no': '1', 'content': content.strip()})

    return paragraphs


def parse_article_info(text):
    """조문 텍스트에서 조번호와 조제목 추출

    예: "제1조(목적) 이 법은..." → article_no="1", article_title="제1조(목적)"
    """
    # 제N조 또는 제N조의N 패턴
    match = re.match(r'(제(\d+)조(?:의(\d+))?(?:\([^)]+\))?)', text)
    if match:
        article_title = match.group(1)
        article_no = match.group(2)
        if match.group(3):
            article_no += f"의{match.group(3)}"
        return article_no, article_title
    return None, None


def save_incremental(data, category, output_dir=RAW_DIR):
    """법령 하나가 크롤링될 때마다 즉시 저장 (실시간 업데이트)"""
    filename = os.path.join(output_dir, f"rd_{category}.json")

    # 기존 파일이 있으면 로드, 없으면 새 리스트
    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except:
            existing_data = []

    # 중복 체크 (같은 law_id가 있으면 업데이트)
    law_id = data['meta_info']['law_id']
    found = False
    for i, item in enumerate(existing_data):
        if item.get('meta_info', {}).get('law_id') == law_id:
            existing_data[i] = data
            found = True
            break

    if not found:
        existing_data.append(data)

    # 저장
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


def get_law_details_with_playwright(laws, max_count=5, incremental_save=True):
    """Playwright로 상세 페이지 크롤링 (JS 렌더링 필요)

    새로운 데이터 구조:
    - meta_info: 법령 메타정보 (시행일, 공포일, 개정유형 등)
    - body: 본조항 리스트 (항 분리 포함)
    - addenda: 부칙 리스트
    - tables: 별표 리스트 (HTML 형식)

    Args:
        incremental_save: True면 법령 하나마다 즉시 파일 저장
    """
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

                # 1. ct_sub에서 메타정보 추출
                meta_text = page.evaluate('''() => {
                    const ctSub = document.querySelector('.ct_sub');
                    return ctSub ? ctSub.innerText.trim() : '';
                }''')

                meta_info = parse_meta_info(meta_text, law)

                # 1.5. 부모법 ID 추출 (ALLJO 태그 중 텍스트가 "법"인 경우만)
                parent_law_id = page.evaluate('''() => {
                    const alljoLinks = document.querySelectorAll('a[onclick*="ALLJO"]');
                    for (const link of alljoLinks) {
                        const text = link.innerText.trim();
                        if (text === '법') {
                            const match = link.getAttribute('onclick').match(/fncLsLawPop\\('(\\d+)','ALLJO'/);
                            return match ? match[1] : null;
                        }
                    }
                    return null;
                }''')
                meta_info['parent_law_id'] = parent_law_id

                # 현재 법령 ID (내부/외부 참조 구분용)
                current_law_id = law['lsi_seq']

                # 2. 본조항 크롤링 (부칙/별표 제외, 참조 정보 포함)
                body_articles = page.evaluate('''(currentLawId) => {
                    const results = [];
                    const conScroll = document.querySelector('#conScroll');
                    if (!conScroll) return results;
                    
                    const pgroups = conScroll.querySelectorAll('.pgroup');
                    pgroups.forEach((el, idx) => {
                        // 부칙 영역인지 확인
                        const isAddendum = el.closest('#arDivArea') !== null;
                        // 별표 영역인지 확인 (별표는 별도 처리)
                        const text = el.innerText.trim();
                        const isTable = text.startsWith('별표') || text.startsWith('[별표');
                        
                        if (!isAddendum && !isTable && text.length > 5) {
                            // 참조 정보 추출
                            const references = [];
                            const refLinks = el.querySelectorAll('a[onclick*="fncLsLawPop"]');
                            refLinks.forEach(link => {
                                const onclick = link.getAttribute('onclick') || '';
                                const match = onclick.match(/fncLsLawPop\\('(\\d+)','(\\w+)'/);
                                if (match) {
                                    const targetId = match[1];
                                    const refType = match[2];  // ALLJO or JO
                                    const refText = link.innerText.trim();
                                    
                                    references.push({
                                        ref_text: refText,
                                        target_id: targetId,
                                        ref_type: refType,  // ALLJO=법률전체, JO=조문
                                        type: (refType === 'JO' || refType === 'ALLJO') ? 
                                              (targetId.substring(0, 7) === currentLawId.substring(0, 7) ? 'internal' : 'external') 
                                              : 'external'
                                    });
                                }
                            });
                            
                            results.push({
                                text: text,
                                html: el.innerHTML,  // HTML도 저장 (참조 링크 포함)
                                index: idx,
                                references: references
                            });
                        }
                    });
                    return results;
                }''', current_law_id)

                # 3. 부칙 크롤링
                addenda_items = page.evaluate('''() => {
                    const results = [];
                    const arDivArea = document.querySelector('#arDivArea');
                    if (!arDivArea) return results;
                    
                    const pgroups = arDivArea.querySelectorAll('.pgroup');
                    pgroups.forEach((el, idx) => {
                        const text = el.innerText.trim();
                        if (text.length > 5) {
                            results.push({
                                text: text,
                                index: idx
                            });
                        }
                    });
                    return results;
                }''')

                # 4. 별표 크롤링 (HTML 형식 + 다운로드 링크)
                # 별표는 ul.pconfile > li > span.pcf_cover 구조 안에 있음
                table_items = page.evaluate('''() => {
                    const results = [];
                    
                    // 방법 1: ul.pconfile 내의 별표 찾기 (정확한 구조)
                    const pconfile = document.querySelector('ul.pconfile');
                    if (pconfile) {
                        const listItems = pconfile.querySelectorAll('li');
                        listItems.forEach(li => {
                            const titleLink = li.querySelector('a.blu[onclick*="bylInfoDiv"]');
                            if (titleLink) {
                                const title = titleLink.innerText.trim();
                                
                                // 다운로드 링크 찾기
                                const downloadLinks = [];
                                const dlLinks = li.querySelectorAll('a[href*="flDownload.do"]');
                                dlLinks.forEach(dl => {
                                    downloadLinks.push({
                                        href: dl.href,
                                        text: dl.innerText.trim() || '다운로드'
                                    });
                                });
                                
                                // 별표 내용이 있는 div 찾기 (svBy로 시작하는 id)
                                const onclickVal = titleLink.getAttribute('onclick') || '';
                                const idMatch = onclickVal.match(/bylInfoDiv\\('(\\d+)'\\)/);
                                let contentHtml = '';
                                if (idMatch) {
                                    const contentDiv = document.querySelector('#svBy' + idMatch[1]);
                                    if (contentDiv) {
                                        contentHtml = contentDiv.innerHTML;
                                    }
                                }
                                
                                results.push({
                                    title: title,
                                    html: contentHtml || li.outerHTML,
                                    downloadLinks: downloadLinks
                                });
                            }
                        });
                    }
                    
                    // 방법 2: pconfile이 없는 경우 다른 방식 시도
                    if (results.length === 0) {
                        const conScroll = document.querySelector('#conScroll');
                        if (conScroll) {
                            const bylLinks = conScroll.querySelectorAll('a.blu[onclick*="bylInfoDiv"]');
                            bylLinks.forEach(link => {
                                const title = link.innerText.trim();
                                // li 또는 가장 가까운 컨테이너 찾기
                                const parent = link.closest('li') || link.closest('span.pcf_cover') || link.parentElement;
                                
                                const downloadLinks = [];
                                if (parent) {
                                    const dlLinks = parent.querySelectorAll('a[href*="flDownload.do"]');
                                    dlLinks.forEach(dl => {
                                        downloadLinks.push({
                                            href: dl.href,
                                            text: dl.innerText.trim() || '다운로드'
                                        });
                                    });
                                }
                                
                                const onclickVal = link.getAttribute('onclick') || '';
                                const idMatch = onclickVal.match(/bylInfoDiv\\('(\\d+)'\\)/);
                                let contentHtml = '';
                                if (idMatch) {
                                    const contentDiv = document.querySelector('#svBy' + idMatch[1]);
                                    if (contentDiv) {
                                        contentHtml = contentDiv.innerHTML;
                                    }
                                }
                                
                                results.push({
                                    title: title,
                                    html: contentHtml || (parent ? parent.outerHTML : ''),
                                    downloadLinks: downloadLinks
                                });
                            });
                        }
                    }
                    
                    return results;
                }''')

                # 본조항 파싱 (참조 정보 포함)
                body = []
                for item in body_articles:
                    text = item['text']
                    article_no, article_title = parse_article_info(text)

                    if article_no:
                        paragraphs = parse_paragraphs(text)
                        body.append({
                            'article_no': article_no,
                            'article_title': article_title,
                            'article_text_full': text,
                            'paragraphs': paragraphs,
                            # 참조 정보 추가
                            'references': item.get('references', [])
                        })

                # 부칙 파싱
                addenda = []
                for item in addenda_items:
                    text = item['text']
                    # 부칙 제목 추출: "부      칙 <대통령령 제35947호, 2025. 12. 30.>"
                    title_match = re.match(r'(부\s*칙\s*<[^>]+>)', text)
                    title = title_match.group(
                        1) if title_match else f"부칙 {len(addenda) + 1}"
                    addenda.append({
                        'article_title': title,
                        'content': text
                    })

                # 별표 파싱 (다운로드 링크 포함)
                tables = []
                for item in table_items:
                    tables.append({
                        'article_title': item.get('title', ''),
                        'content_html': item.get('html', ''),
                        'download_links': item.get('downloadLinks', [])
                    })

                # 최종 데이터 구조
                law_data = {
                    'meta_info': meta_info,
                    'body': body,
                    'addenda': addenda,
                    'tables': tables
                }

                results.append(law_data)
                print(
                    f"  → 본조항 {len(body)}개, 부칙 {len(addenda)}개, 별표 {len(tables)}개 수집 완료")

                # 실시간 저장 (법령 하나 크롤링될 때마다)
                if incremental_save:
                    category = meta_info.get('category_main', 'unknown')
                    save_incremental(law_data, category)
                    print(f"  → rd_{category}.json 업데이트 완료")

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

    # Step 0: 웹사이트에서 모든 카테고리 추출
    categories = extract_all_categories()
    print(f"\n크롤링 대상: {len(categories)}개 분야")

    # 추출된 카테고리 목록 출력
    print("\n[추출된 카테고리 목록]")
    for i, cat in enumerate(categories[:10]):  # 처음 10개만 출력
        print(f"  {i+1}. {cat['name']}")
    if len(categories) > 10:
        print(f"  ... 외 {len(categories) - 10}개")
    print()

    # Step 1: 모든 분야의 법령 목록 가져오기
    print("[Step 1] 모든 분야 법령 목록 조회 중...")

    all_laws = []
    for i, category in enumerate(categories):
        print(f"\n  [{i+1}/{len(categories)}] {category['name']} 분야 크롤링...")
        laws = get_all_law_list_for_category(category)
        all_laws.extend(laws)
        print(f"  → {category['name']} 완료: {len(laws)}개 법령")

    print(f"\n총 {len(all_laws)}개 법령 발견\n")

    laws = all_laws  # 변수명 통일

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

        # 분야별 별도 파일 저장 (실시간 저장으로 이미 생성되었지만 최종 확인)
        for cat in categories:
            cat_results = [r for r in results if r['meta_info']
                           ['category_main'] == cat['name']]
            if cat_results:
                filename = os.path.join(RAW_DIR, f"rd_{cat['name']}.json")
                save_results(cat_results, filename)

        print(f"\n=== 크롤링 완료 ===")
        total_body = sum(len(r['body']) for r in results)
        total_addenda = sum(len(r['addenda']) for r in results)
        total_tables = sum(len(r['tables']) for r in results)
        print(
            f"총 {len(results)}개 법령, {total_body}개 본조항, {total_addenda}개 부칙, {total_tables}개 별표 수집")

        # 분야별 통계
        for cat in categories:
            cat_results = [r for r in results if r['meta_info']
                           ['category_main'] == cat['name']]
            cat_count = len(cat_results)
            if cat_count > 0:
                cat_body = sum(len(r['body']) for r in cat_results)
                cat_addenda = sum(len(r['addenda']) for r in cat_results)
                print(
                    f"  - {cat['name']}: {cat_count}개 법령, {cat_body}개 조항, {cat_addenda}개 부칙 → rd_{cat['name']}.json")
    else:
        print("크롤링된 데이터가 없습니다.")


if __name__ == "__main__":
    main()
