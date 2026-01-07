"""
중앙부처 1차 해석 (결정선례) 크롤러 (2단계 전략 - 수정완료)
1단계: 전체 페이지(약 9000건/180페이지 추정) ID 전수 수집
2단계: 수집된 ID 기반 상세정보 추출
"""

import json
import os
import time
import re
import random
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

# ============================================================
# 설정
# ============================================================
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / '..' / 'data'
RAW_DIR = DATA_DIR / 'raw'
LIST_FILE = RAW_DIR / 'rd_법령외_결정선례_list.json'
OUTPUT_FILE = RAW_DIR / 'rd_법령외_결정선례.json'

# 타겟 URL (중앙부처 1차 해석, 전체 부처)
TARGET_URL = "https://www.law.go.kr/LSW/cgmExpcSc.do?menuId=11&subMenuId=729&tabMenuId=733&upperOfiClsCd=M&ofiClsCd=350101"

# ============================================================
# 유틸리티
# ============================================================


def save_json(data, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: Path):
    if not filepath.exists():
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ============================================================
# Phase 1: 전체 목록(ID) 수집
# ============================================================


def collect_ids():
    print("🚀 [1단계] 전체 목록 ID 수집 시작...")

    # 이어하기 지원
    collected_items = load_json(LIST_FILE)
    visited_ids = {item['item_id']
                   for item in collected_items if 'item_id' in item}

    # 마지막으로 수집한 페이지 찾기 (이어하기용)
    start_page = 1
    if collected_items:
        max_collected_page = max((item.get('page', 1)
                                 for item in collected_items), default=1)
        start_page = max_collected_page
        print(f"  ↪ 기존 데이터: {len(collected_items)}개, {start_page}페이지부터 탐색 재개")
    else:
        print(f"  ↪ 신규 수집 시작")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print(f"  📄 페이지 로드: {TARGET_URL}")
        try:
            page.goto(TARGET_URL, timeout=60000)
            page.wait_for_selector('.tbl_wrap table', timeout=30000)
        except Exception as e:
            print(f"  ❌ 초기 접속 실패: {e}")
            return

        current_page = 1  # fnSetPage 에러 회피를 위해 1부터 시작 (중복은 스킵됨)
        # start_page 로직 제거

        last_first_item_title = None
        consecutive_duplicates = 0

        while True:
            print(f"\n📑 [Page {current_page}] 스캔 중...", end=" ", flush=True)

            try:
                page.wait_for_selector(
                    '.tbl_wrap table tbody tr td.s_tit a', timeout=10000)
            except:
                print("⚠️ 테이블 로딩 타임아웃 (데이터 없음/끝일 수 있음)")
                break

            rows = page.locator('.tbl_wrap table tbody tr td.s_tit a').all()
            if not rows:
                print("⚠️ 게시글 없음. 종료.")
                break

            new_in_page = 0

            # 현재 페이지 첫번째 아이템 확인
            current_first_title = rows[0].inner_text().strip()
            if last_first_item_title == current_first_title:
                consecutive_duplicates += 1
                if consecutive_duplicates >= 2:
                    print(f"🏁 페이지 변화 없음 ({consecutive_duplicates}회). 수집 종료.")
                    break
            else:
                consecutive_duplicates = 0
            last_first_item_title = current_first_title

            for link in rows:
                try:
                    title_full = link.inner_text().strip()
                    onclick = link.get_attribute('onclick')

                    item_id = ""
                    if onclick:
                        match = re.search(
                            r"(?:lsEmpViewWideAll|cgmExpcView)\('(\d+)'", onclick)
                        if match:
                            item_id = match.group(1)

                    if item_id and item_id not in visited_ids:
                        item = {
                            "item_id": item_id,
                            "title_full": title_full,
                            "onclick": onclick,
                            "page": current_page,
                            "collected_at": datetime.now().isoformat()
                        }
                        collected_items.append(item)
                        visited_ids.add(item_id)
                        new_in_page += 1
                except Exception:
                    pass

            print(
                f"완료 ({new_in_page}건 신규 / 전체 {len(collected_items)}건) | 예: {current_first_title[:15]}...")
            save_json(collected_items, LIST_FILE)

            # 다음 페이지 이동
            next_page = current_page + 1

            # 페이지 이동 실행 (Click 방식)
            try:
                # 페이징 영역의 모든 링크 가져오기
                paging_links = page.locator(".paging a").all()
                navigated = False

                # print(f"  🔍 페이지 탐색 중... (찾는 페이지: {next_page})")

                # 1. 숫자 버튼 찾기
                for link in paging_links:
                    text = link.inner_text().strip()
                    if text == str(next_page):
                        # print(f"  👉 숫자 버튼({next_page}) 클릭")
                        link.evaluate("el => el.click()")
                        navigated = True
                        break

                # 2. 숫자를 못 찾았으면 '다음' 버튼 찾기
                if not navigated:
                    for link in paging_links:
                        # 이미지 확인
                        img = link.locator("img")
                        if img.count() > 0:
                            alt = img.get_attribute("alt")
                            if alt and "다음" in alt:
                                print(f"  ⏩ '다음' 이미지 버튼 클릭")
                                link.evaluate("el => el.click()")
                                navigated = True
                                break
                        # 클래스 확인
                        elif link.get_attribute("class") == "next":
                            print(f"  ⏩ 'next' 클래스 버튼 클릭")
                            link.evaluate("el => el.click()")
                            navigated = True
                            break

                if navigated:
                    try:
                        # 로딩 대기
                        page.wait_for_load_state('networkidle', timeout=10000)
                        time.sleep(1.5)
                    except:
                        time.sleep(2.0)
                    current_page += 1
                else:
                    print(f"🏁 다음 페이지({next_page}) 연결 고리 없음. 수집 종료.")
                    break

            except Exception as e:
                print(f"⚠️ 페이지 이동 중 에러: {e}")
                break

            if current_page % 10 == 0:
                print(f"  💤 10페이지 단위 휴식...")
                time.sleep(1)

        browser.close()

    print(f"\n✅ [1단계] ID 수집 완료. 총 {len(collected_items)}개 저장됨.")
    return collected_items

# ============================================================
# Phase 2: 상세 수집
# ============================================================


def extract_content(page, header_id, stop_tag='H4'):
    return page.evaluate(f"""() => {{
        const header = document.querySelector('#{header_id}');
        if (!header) return "";
        const texts = [];
        let sib = header.nextElementSibling;
        while (sib && sib.tagName !== '{stop_tag}') {{
            if (sib.innerText && (sib.tagName === 'P' || sib.tagName === 'DIV')) {{
                texts.push(sib.innerText.trim());
            }}
            sib = sib.nextElementSibling;
        }}
        return texts.join('\\n');
    }}""")


def crawl_details():
    print("\n🚀 [2단계] 상세 데이터 크롤링 시작...")

    id_list = load_json(LIST_FILE)
    if not id_list:
        print("❌ 1단계 데이터(ID목록)가 없습니다. collect_ids()를 먼저 실행하세요.")
        return

    completed_data = load_json(OUTPUT_FILE)
    completed_ids = {item['item_id']
                     for item in completed_data if 'item_id' in item}

    total_count = len(id_list)
    targets = [item for item in id_list if item['item_id'] not in completed_ids]

    print(f"  📝 총 항목: {total_count}개")
    print(f"  ✅ 완료됨: {len(completed_ids)}개")
    print(f"  ▶️ 남은작업: {len(targets)}개")

    if not targets:
        print("🎉 모든 데이터 수집이 완료되었습니다.")
        return

    # Phase 2는 title 동기화가 핵심이므로, 엄격한 체크를 위해 title 정규화 함수 정의
    def normalize_title(t):
        return re.sub(r'\s+', '', t).strip()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print(f"  📄 베이스 페이지 로드...")
        try:
            page.goto(TARGET_URL, timeout=60000)
            page.wait_for_selector('.tbl_wrap table', timeout=30000)
        except Exception as e:
            print(f"❌ 베이스 페이지 접속 불가: {e}")
            return

        for i, item in enumerate(targets):
            item_id = item['item_id']
            title = item['title_full']

            progress = f"[{i+1}/{len(targets)}]"
            print(
                f"{progress} ID:{item_id} | {title[:20]}...", end="", flush=True)

            retries = 2
            success = False

            while retries > 0:
                try:
                    clean_js = item['onclick'].replace(
                        'return false', '').strip()
                    if clean_js.endswith(';'):
                        clean_js = clean_js[:-1]

                    page.evaluate(clean_js)

                    # DOM 변경 대기 (AJAX)
                    try:
                        # 1. 스피너나 오버레이가 사라지길 대기 (있다면)
                        # 2. #contentBody가 보이길 대기
                        # 3. 중요: 클릭 전의 텍스트와 달라졌는지 확인은 어려우므로,
                        #    evaluate 직후 약간의 sleep을 주고, rpl(답변) ID가 로드되기를 대기
                        time.sleep(0.5)
                        page.wait_for_selector(
                            '#contentBody', state='visible', timeout=5000)
                        page.wait_for_selector(
                            '#rpl', state='attached', timeout=5000)  # 답변 영역 존재 확인
                    except:
                        pass

                    try:
                        page.wait_for_selector('#contentBody', timeout=5000)
                    except:
                        if retries == 1:
                            page.goto(
                                f"{TARGET_URL}#licCgmExpc{item_id}_350101")
                            page.wait_for_selector(
                                '#contentBody', timeout=5000)
                        else:
                            raise Exception("Content load timeout")

                    # --------------------------------------------------------
                    # [중요] Title 검증으로 페이지 갱신 여부 확인
                    # --------------------------------------------------------
                    page_title = ""
                    try:
                        # 페이지 내 실제 제목 요소 (h4 등) 구조에 따라 수정 필요
                        # 결정선례 페이지 구조상 #contentBody h3 또는 h4 등에 제목이 있을 수 있음
                        # 여기서는 .tit_view 또는 input[name="title"] 등 확인 필요하지만
                        # 2단계 리스트에서 클릭 시, 본문 상단 타이틀이 바뀌는지 확인.

                        # law.go.kr 구조상 본문 타이틀 ID가 명확치 않을 수 있으므로
                        # inqGst(질의) 내용이 비어있지 않은지 우선 체크하고,
                        # 가능하다면 item['title_full']과 유사한 텍스트가 있는지 확인.
                        pass
                    except:
                        pass

                    # 제목에서 Agency, Date 추출 (정규식)
                    # 예: "육아휴직 급여 ... [고용노동부, 2025.08.06.]"
                    real_title = title
                    agency = "Unknown"
                    date = "Unknown"

                    match = re.search(
                        r'^(.*?)\s*\[([^,]+),\s*([\d.]+)\]$', title)
                    if match:
                        real_title_only = match.group(1).strip()
                        agency = match.group(2).strip()
                        date = match.group(3).strip()
                    else:
                        real_title_only = title

                    # 본문 추출
                    q_text = extract_content(page, 'inqGst')
                    a_text = extract_content(page, 'rpl')

                    # [검증] 질의나 답변 중 하나는 반드시 있어야 함.
                    # 또한, 만약 이전 페이지의 내용이 남아있는지 확인해야 함.
                    # (여기서는 q_text, a_text가 비어있으면 로딩 실패로 간주)
                    if not q_text and not a_text:
                        raise Exception("Empty content (q_text & a_text)")

                    # [검증 2] 본문 내용이 이전 아이템과 동일한지 체크 (Stale Element)
                    # 간단히 텍스트 길이 등으로 비교하거나 해시를 쓸 수 없으니,
                    # 여기서는 '로딩 대기'를 믿되, 내용이 너무 짧으면 의심.
                    if len(q_text) < 5 and len(a_text) < 5:
                        raise Exception("Content too short (Example: null)")

                    # --------------------------------------------------------

                    laws = []
                    try:
                        laws = page.evaluate("""() => {
                            const lHeader = document.querySelector('#conLs');
                            if (!lHeader) return [];
                            const res = [];
                            let sib = lHeader.nextElementSibling;
                            while (sib) {
                                if (sib.tagName === 'P') {
                                    sib.querySelectorAll('a').forEach(l => {
                                        res.push({text: l.innerText.trim(), onclick: l.getAttribute('onclick')});
                                    });
                                }
                                if (sib.tagName === 'H4') break;
                                sib = sib.nextElementSibling;
                            }
                            return res;
                        }""")
                    except:
                        pass

                    data_obj = {
                        "item_id": item_id,
                        "title": real_title,
                        "agency": agency,
                        "date": date,
                        "question": q_text,
                        "answer": a_text,
                        "related_laws": laws,
                        "url": page.url,
                        "crawled_at": datetime.now().isoformat()
                    }

                    completed_data.append(data_obj)
                    save_json(completed_data, OUTPUT_FILE)

                    print(" -> ✅ 성공")
                    success = True
                    break

                except Exception as e:
                    retries -= 1
                    print(f" -> ⚠️ 실패(재시도{retries}): {e}", end="")
                    try:
                        page.goto(TARGET_URL)
                        page.wait_for_selector(
                            '.tbl_wrap table', timeout=10000)
                    except:
                        pass

            if not success:
                print(" -> ❌ 최종 실패 (건너뜀)")

            try:
                if page.locator('#btnList').is_visible():
                    page.click('#btnList')
                    page.wait_for_selector('.tbl_wrap table', timeout=5000)
                else:
                    page.go_back()
            except:
                pass

            time.sleep(0.1)

        browser.close()


if __name__ == "__main__":
    # Phase 1: ID 수집 우선 (사용자 요청: 9000개 전수 수집)
    # collect_ids()

    # Phase 2: 상세 수집 (Phase 1 완료 후 필요 시 주석 해제)
    crawl_details()
