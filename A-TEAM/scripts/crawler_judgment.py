from playwright.sync_api import sync_playwright
import json
import time
import re
import signal
import sys

def parse_detail_page(page, detail_url):
    """상세 페이지에서 필요한 정보를 파싱합니다."""
    try:
        page.goto(detail_url, wait_until='networkidle', timeout=30000)
        time.sleep(1)  # 추가 렌더링 대기
        
        data = {}
        
        # JavaScript를 사용하여 th 태그의 텍스트로 인접한 td 값을 찾습니다
        data['자료구분'] = page.evaluate('''() => {
            const th = Array.from(document.querySelectorAll('th')).find(el => el.innerText.includes('자료구분'));
            return th ? th.nextElementSibling?.innerText.trim() : '';
        }''')
        
        data['담당부서'] = page.evaluate('''() => {
            const th = Array.from(document.querySelectorAll('th')).find(el => el.innerText.includes('담당부서'));
            return th ? th.nextElementSibling?.innerText.trim() : '';
        }''')
        
        data['등록일'] = page.evaluate('''() => {
            const th = Array.from(document.querySelectorAll('th')).find(el => el.innerText.includes('등록일'));
            return th ? th.nextElementSibling?.innerText.trim() : '';
        }''')
        
        data['판정사항'] = page.evaluate('''() => {
            const th = Array.from(document.querySelectorAll('th')).find(el => el.innerText.includes('판정사항'));
            return th ? th.nextElementSibling?.innerText.trim() : '';
        }''')
        
        data['판정요지'] = page.evaluate('''() => {
            const th = Array.from(document.querySelectorAll('th')).find(el => el.innerText.includes('판정요지'));
            return th ? th.nextElementSibling?.innerText.trim() : '';
        }''')
        
        return data

    except Exception as e:
        print(f"상세 페이지 파싱 중 에러: {e}")
        return None

def main():
    base_url = "https://nlrc.go.kr/nlrc/mainCase/judgment/index.do"
    detail_base_url = "https://nlrc.go.kr/nlrc/mainCase/judgment/detail.do"
    
    results = []
    target_start = 386
    target_end = 1
    
    print(f"크롤링 시작: {target_start}번 게시물부터 {target_end}번 게시물까지 수집합니다.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        is_crawling = True
        page_num = 1
        empty_page_count = 0  # 연속으로 유효한 게시물이 없는 페이지 수
        max_empty_pages = 3   # 최대 허용 빈 페이지 수
        
        while is_crawling:
            print(f"\n페이지 {page_num} 조회 중...")
            url = f"{base_url}?pageIndex={page_num}"
            
            try:
                page.goto(url, wait_until='networkidle', timeout=30000)
                time.sleep(2)  # 추가 렌더링 대기
                
                # JavaScript로 테이블 행 데이터 추출
                rows_data = page.evaluate('''() => {
                    const rows = Array.from(document.querySelectorAll('tbody tr'));
                    return rows.map(row => {
                        const numTd = row.querySelector('td:nth-child(1)');
                        const titleTd = row.querySelector('td.left, td:nth-child(2)');
                        
                        return {
                            num: numTd ? numTd.innerText.trim() : '',
                            jgmtSn: row.getAttribute('data-jgmt-sn'),
                            jgmtDcsnSeCd: row.getAttribute('data-jgmt-dcsn-se-cd'),
                            title: titleTd ? titleTd.innerText.trim() : ''
                        };
                    });
                }''')
                
                if not rows_data or len(rows_data) == 0:
                    print("게시물이 더 이상 없습니다.")
                    break
                
                print(f"  → {len(rows_data)}개 게시물 발견")
                
                page_processed_count = 0  # 이 페이지에서 처리한 게시물 수
                
                for row_data in rows_data:
                    row_num_text = row_data['num']
                    
                    # 공지사항 등 번호가 숫자가 아닌 경우 건너뜀
                    if not row_num_text.isdigit():
                        continue
                    
                    row_num = int(row_num_text)
                    
                    # 범위 체크
                    if row_num > target_start:
                        continue
                    
                    if row_num < target_end:
                        print(f"목표 번호({target_end}) 미만 도달 ({row_num}). 크롤링 종료.")
                        is_crawling = False
                        break
                    
                    # 타겟 범위 내의 게시물인 경우
                    jgmt_sn = row_data['jgmtSn']
                    jgmt_dcsn_se_cd = row_data['jgmtDcsnSeCd']
                    
                    if not jgmt_sn:
                        print(f"게시물 {row_num}의 ID를 찾을 수 없습니다.")
                        continue
                    
                    detail_url = f"{detail_base_url}?jgmtSn={jgmt_sn}&jgmtDcsnSeCd={jgmt_dcsn_se_cd}"
                    print(f"  [{row_num}번] 상세 데이터 수집 중...")
                    
                    detail_data = parse_detail_page(page, detail_url)
                    
                    if detail_data:
                        detail_data['번호'] = row_num
                        detail_data['제목'] = row_data['title']
                        results.append(detail_data)
                        print(f"    ✓ 수집 완료 (총 {len(results)}건)")
                        page_processed_count += 1
                    
                    # 서버 부하 방지
                    time.sleep(0.5)
                
                # 이 페이지에서 처리한 게시물이 있으면 카운터 리셋, 없으면 증가
                if page_processed_count > 0:
                    empty_page_count = 0
                    print(f"  → 이 페이지에서 {page_processed_count}건 처리됨")
                else:
                    empty_page_count += 1
                    print(f"  → 이 페이지에서 처리된 게시물 없음 (연속 {empty_page_count}페이지)")
                    
                    if empty_page_count >= max_empty_pages:
                        print(f"\n{max_empty_pages}페이지 연속으로 유효한 게시물이 없어 크롤링을 종료합니다.")
                        break
                
                if not is_crawling:
                    break
                    
                page_num += 1
                time.sleep(1)
                
            except Exception as e:
                print(f"페이지 순회 중 에러 발생: {e}")
                break
        
        browser.close()

    # 결과 저장
    if results:
        save_filename = 'judgment_data_jwy.json'
        with open(save_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*50}")
        print(f"총 {len(results)}건의 데이터가 {save_filename}에 저장되었습니다.")
        print(f"{'='*50}")
    else:
        print("\n수집된 데이터가 없습니다.")

if __name__ == "__main__":
    main()
