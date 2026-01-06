"""
노동고용부 행정해석(질의회시) 크롤러
URL: https://www.moel.go.kr/info/publicdata/qnrinfo/qnrInfoList.do
"""

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import json
import time
import re
from datetime import datetime

# 세션
session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

BASE_URL = "https://www.moel.go.kr"
LIST_URL = "https://www.moel.go.kr/info/publicdata/qnrinfo/qnrInfoList.do"


def clean_text(text):
    """
    질문/답변 텍스트 전처리
    - HTML 태그 제거
    - 특수 기호 정리 (〇, ○, ㅇ, >, - 등)
    - 연속 공백/줄바꿈 정리
    """
    if not text:
        return ""

    # HTML 태그 제거
    text = re.sub(r'<BR\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)

    # HTML 엔티티 변환
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&amp;', '&').replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"').replace('&#39;', "'")

    # 특수 기호 정리 (줄 시작의 불릿 포인트들)
    # 〇, ○, ㅇ, ◦, ●, · 등을 일관되게 처리
    text = re.sub(r'^[\s]*[〇○ㅇ◦●·▶►◇◆■□▪▫]\s*', '• ', text, flags=re.MULTILINE)

    # > 기호 (인용 표시) 정리
    text = re.sub(r'^[\s]*>\s*', '', text, flags=re.MULTILINE)

    # - 로 시작하는 줄 정리
    text = re.sub(r'^[\s]*-\s+', '- ', text, flags=re.MULTILINE)

    # 연속 줄바꿈 정리 (3개 이상 → 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 연속 공백 정리
    text = re.sub(r'[ \t]+', ' ', text)

    # 줄 앞뒤 공백 정리
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def get_post_list_page(page=1):
    """게시글 목록 페이지 가져오기 (GET with pageIndex)"""
    params = {"pageIndex": page}

    try:
        response = session.get(LIST_URL, headers=headers, params=params)
        response.encoding = 'utf-8'

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # 테이블 tbody 내의 tr들
            rows = soup.select('.tstyle_list tbody tr')

            posts = []
            for row in rows:
                cols = row.select('td')
                if len(cols) >= 5:
                    # 링크 추출
                    link = row.select_one('a.ellipsis')
                    if link:
                        href = link.get('href', '')
                        title = link.get(
                            'title', '') or link.get_text(strip=True)

                        posts.append({
                            'number': cols[0].get('title', '') or cols[0].get_text(strip=True),
                            'title': title,
                            'url': BASE_URL + href if href.startswith('/') else href,
                            # 담당부서
                            'department': cols[2].get('title', '') or cols[2].get_text(strip=True),
                            # 담당자
                            'person': cols[3].get('title', '') or cols[3].get_text(strip=True),
                            # 게시일
                            'reg_date': cols[4].get('title', '') or cols[4].get_text(strip=True),
                            # 조회수
                            'views': cols[5].get('title', '') if len(cols) > 5 else '',
                        })

            return posts
    except Exception as e:
        print(f"목록 조회 에러 (페이지 {page}): {e}")

    return []


def get_post_detail(url):
    """게시글 상세 내용 가져오기"""
    try:
        response = session.get(url, headers=headers)
        response.encoding = 'utf-8'

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # 본문 영역 찾기 (여러 셀렉터 시도)
            content_area = (
                soup.select_one('.view_cont') or
                soup.select_one('.board_view') or
                soup.select_one('#contents') or
                soup.select_one('.content')
            )

            raw_content = content_area.get_text(
                strip=False) if content_area else ''
            raw_html = str(content_area) if content_area else ''

            return {
                'raw_content': raw_content.strip(),
                'raw_html': raw_html,
            }
    except Exception as e:
        print(f"상세 조회 에러 ({url}): {e}")

    return None


def parse_qa_content(raw_content):
    """다양한 형식의 질의/답변 파싱"""

    result = {
        'questions': [],
        'answers': [],
        'parse_success': False,
        'format_detected': None,
    }

    if not raw_content:
        return result

    # 패턴 1: <질 의>, <답 변>, <회 시> 형식
    pattern1_q = re.search(
        r'<\s*질\s*의?\s*>(.+?)(?=<\s*(?:답|회)\s*)', raw_content, re.DOTALL)
    pattern1_a = re.search(
        r'<\s*(?:답\s*변|회\s*시)\s*>(.+?)(?=<|$)', raw_content, re.DOTALL)

    if pattern1_q and pattern1_a:
        result['questions'].append(pattern1_q.group(1).strip())
        result['answers'].append(pattern1_a.group(1).strip())
        result['parse_success'] = True
        result['format_detected'] = 'angle_bracket'
        return result

    # 패턴 2: [질의], [답변], [회시] 형식
    pattern2_q = re.search(
        r'\[\s*질의\s*\](.+?)(?=\[\s*(?:답변|회시)\s*\])', raw_content, re.DOTALL)
    pattern2_a = re.search(
        r'\[\s*(?:답변|회시)\s*\](.+?)(?=\[|$)', raw_content, re.DOTALL)

    if pattern2_q and pattern2_a:
        result['questions'].append(pattern2_q.group(1).strip())
        result['answers'].append(pattern2_a.group(1).strip())
        result['parse_success'] = True
        result['format_detected'] = 'square_bracket'
        return result

    # 패턴 3: □ 질의1 / □ 답변1, □질의.01 / □답변.01 형식 (새로 추가)
    # □ 질의1, □ 질의.01, □질의1, □질의.01 등 다양한 변형
    box_qs = re.findall(
        r'□\s*질의[\s.]*\d*\s*\n?(.*?)(?=□\s*(?:답변|질의)[\s.]*\d*|$)', raw_content, re.DOTALL)
    box_as = re.findall(
        r'□\s*답변[\s.]*\d*\s*\n?(.*?)(?=□\s*(?:답변|질의)[\s.]*\d*|$)', raw_content, re.DOTALL)

    if box_qs and box_as:
        result['questions'] = [q.strip() for q in box_qs if q.strip()]
        result['answers'] = [a.strip() for a in box_as if a.strip()]
        result['parse_success'] = True
        result['format_detected'] = 'box_numbered'
        return result

    # 패턴 4: □ 질의내용 / □ 답변내용 형식
    box_q_content = re.search(
        r'□\s*질의\s*내용\s*\n?(.*?)(?=□\s*답변|$)', raw_content, re.DOTALL)
    box_a_content = re.search(
        r'□\s*답변\s*내용\s*\n?(.*?)(?=□|목록|$)', raw_content, re.DOTALL)

    if box_q_content and box_a_content:
        result['questions'].append(box_q_content.group(1).strip())
        result['answers'].append(box_a_content.group(1).strip())
        result['parse_success'] = True
        result['format_detected'] = 'box_content'
        return result

    # 패턴 5: 질의요지 + (질의1)(질의2) / 답변내용 형식
    if '질의요지' in raw_content or '질의 요지' in raw_content:
        q_section = re.search(r'(?:질의\s*요지)(.+?)(?=답변|회시|$)',
                              raw_content, re.DOTALL | re.IGNORECASE)
        a_section = re.search(r'(?:답변\s*내용|회시\s*내용|답\s*변)(.+?)$',
                              raw_content, re.DOTALL | re.IGNORECASE)

        if q_section:
            q_text = q_section.group(1)
            individual_qs = re.findall(
                r'\(질의\s*\d*\)(.+?)(?=\(질의\s*\d*\)|$)', q_text, re.DOTALL)
            result['questions'] = [q.strip() for q in individual_qs] if individual_qs else [
                q_text.strip()]

        if a_section:
            a_text = a_section.group(1)
            individual_as = re.findall(
                r'\(답변\s*\d*\)(.+?)(?=\(답변\s*\d*\)|$)', a_text, re.DOTALL)
            result['answers'] = [a.strip() for a in individual_as] if individual_as else [
                a_text.strip()]

        if result['questions'] or result['answers']:
            result['parse_success'] = True
            result['format_detected'] = 'numbered'
            return result

    # 패턴 6: ○ (질의N) ... <답변내용> (답변N) 형식
    circle_qs = re.findall(
        r'○\s*\(질의\d*\)\s*(.*?)(?=○\s*\(질의\d*\)|<답변|$)', raw_content, re.DOTALL)
    if '<답변내용>' in raw_content or '<답변 내용>' in raw_content:
        circle_as = re.findall(
            r'\(답변\d*\)\s*(.*?)(?=\(답변\d*\)|$)', raw_content, re.DOTALL)
        if circle_qs and circle_as:
            result['questions'] = [q.strip() for q in circle_qs if q.strip()]
            result['answers'] = [a.strip() for a in circle_as if a.strip()]
            result['parse_success'] = True
            result['format_detected'] = 'circle_numbered'
            return result

    # 패턴 7: <회신 내용> 형식 (질의는 본문에서 추출)
    if '<회신 내용>' in raw_content or '<회신내용>' in raw_content:
        # 회신 내용 앞부분을 질의로 추정
        reply_match = re.search(
            r'<회신\s*내용>(.*?)(?=목록|$)', raw_content, re.DOTALL)
        if reply_match:
            result['answers'].append(reply_match.group(1).strip())
            # 질의는 회신 전 내용에서 추출
            before_reply = raw_content.split('<회신')[0]
            # 마지막 문단을 질의로 사용
            last_paragraph = before_reply.strip().split(
                '\n\n')[-1] if before_reply else ''
            if last_paragraph:
                result['questions'].append(last_paragraph.strip())
            result['parse_success'] = True
            result['format_detected'] = 'reply_format'
            return result

    result['format_detected'] = 'unknown'
    return result


def apply_clean_text_to_result(result):
    """파싱 결과에 clean_text 적용"""
    if result['parse_success']:
        result['questions'] = [clean_text(q) for q in result['questions']]
        result['answers'] = [clean_text(a) for a in result['answers']]
    return result


def crawl_all_posts():
    """모든 페이지 게시글 목록 수집"""
    all_posts = []
    page = 1

    while True:
        print(f"  페이지 {page} 조회 중...")
        posts = get_post_list_page(page)

        if not posts:
            print(f"  → 페이지 {page}에서 결과 없음. 수집 종료.")
            break

        all_posts.extend(posts)
        print(f"  → {len(posts)}개 게시글 수집 (누적: {len(all_posts)}개)")

        page += 1
        time.sleep(0.3)

    return all_posts


def crawl_with_details(posts, save_interval=100):
    """상세 내용까지 포함하여 크롤링"""
    results = []

    for i, post in enumerate(posts):
        print(f"[{i+1}/{len(posts)}] {post['title'][:40]}...")

        detail = get_post_detail(post['url'])

        if detail:
            parsed = parse_qa_content(detail['raw_content'])
            # 질문/답변 텍스트 전처리 적용
            parsed = apply_clean_text_to_result(parsed)

            results.append({
                'number': post['number'],
                'title': post['title'],
                'department': post['department'],
                'person': post.get('person', ''),
                'reg_date': post['reg_date'],
                'views': post.get('views', ''),
                'url': post['url'],
                'raw_content': detail['raw_content'],
                'parsed': parsed,
            })

            status = "✓" if parsed['parse_success'] else "✗"
            print(f"  → {status} {parsed['format_detected']}")
        else:
            print(f"  → ✗ 상세 내용 없음")

        # 중간 저장
        if (i + 1) % save_interval == 0:
            save_results(results, f'행정해석_temp_{i+1}.json')
            print(f"  💾 중간 저장 완료 ({i+1}개)")

        time.sleep(0.3)

    return results


def save_results(data, filename):
    """결과 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장: {filename}")


def main():
    print("=== 노동고용부 행정해석(질의회시) 크롤링 시작 ===\n")

    # Step 1: 게시글 목록 전체 수집
    print("[Step 1] 게시글 목록 수집 중...")
    posts = crawl_all_posts()
    print(f"\n총 {len(posts)}개 게시글 발견\n")

    if not posts:
        print("게시글을 가져오지 못했습니다.")
        return

    # 목록만 먼저 저장 (안전망)
    save_results(posts, '행정해석_목록.json')

    # Step 2: 상세 내용 크롤링 + 파싱
    print("\n[Step 2] 상세 내용 크롤링 및 파싱 중...")
    results = crawl_with_details(posts)

    # Step 3: 최종 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(results, f'행정해석_{timestamp}.json')

    # 통계
    success_count = sum(1 for r in results if r['parsed']['parse_success'])
    print(f"\n=== 크롤링 완료 ===")
    print(f"총 {len(results)}개 게시글")
    print(
        f"파싱 성공: {success_count}개 ({100*success_count/len(results):.1f}%)" if results else "")


if __name__ == "__main__":
    main()
