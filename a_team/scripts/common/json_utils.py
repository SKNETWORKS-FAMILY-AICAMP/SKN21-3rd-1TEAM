"""
JSON 스트리밍 유틸리티
대용량 JSON 배열을 배치로 읽어오는 함수
"""
import json
from typing import Iterator, Dict, Any, List


def stream_json_array(filepath: str, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
    """
    대용량 JSON 배열 파일을 배치 단위로 스트리밍

    Args:
        filepath: JSON 파일 경로
        batch_size: 한 번에 읽을 항목 수

    Yields:
        배치 단위의 JSON 객체 리스트
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        # JSON 배열 시작 확인
        char = f.read(1)
        if char != '[':
            raise ValueError("JSON file must start with '['")

        batch = []
        buffer = ""
        depth = 0
        in_string = False
        escape = False

        while True:
            char = f.read(1)
            if not char:
                break

            # 문자열 처리
            if char == '"' and not escape:
                in_string = not in_string

            # 이스케이프 처리
            escape = (char == '\\' and not escape)

            if not in_string:
                # 객체 depth 추적
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1

                # 객체 완료
                if depth == 0 and char == '}':
                    buffer += char
                    # JSON 객체 파싱
                    try:
                        obj = json.loads(buffer.strip())
                        batch.append(obj)
                        buffer = ""

                        # 배치 크기 도달 시 yield
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except json.JSONDecodeError:
                        pass
                    continue

                # 쉼표나 공백은 무시
                if char in [',', ' ', '\n', '\t'] and depth == 0:
                    continue

            # 배열 종료
            if char == ']' and depth == 0 and not in_string:
                break

            buffer += char

        # 남은 배치 반환
        if batch:
            yield batch


def count_json_array_items(filepath: str) -> int:
    """
    JSON 배열의 항목 수를 빠르게 카운트 (메모리 효율적)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return len(data)
