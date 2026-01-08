"""
노동법 RAG 챗봇 평가 스크립트 (LangGraph V1 버전)

Golden Dataset을 사용하여 LangGraph 기반 RAG 모델(V1)의 성능을 평가합니다.
Ragas 메트릭(Faithfulness, Answer Relevancy, Context Precision/Recall)을 계산합니다.

Usage:
    # 기본 실행
    uv run a_team/scripts/evaluate_rag_V1.py

    # 샘플 수 지정 (테스트용)
    uv run a_team/scripts/evaluate_rag_V1.py --sample 10

    # 커스텀 골든셋 경로
    uv run a_team/scripts/evaluate_rag_other.py --golden-set a_team/data/evaluation/golden_set_quota_20.json
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Ragas 평가 메트릭
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

# LangChain (평가용 LLM)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# 환경변수 로드
_SCRIPT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=_SCRIPT_DIR / ".env")

# 경고 메시지 필터링
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================
# Golden Dataset 로드
# ============================================================
def load_golden_dataset(path: str) -> pd.DataFrame:
    """
    Golden Dataset JSON 파일을 로드합니다.

    Args:
        path: JSON 파일 경로

    Returns:
        DataFrame with columns: user_input, reference (ground truth)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden Dataset을 찾을 수 없습니다: {path}")

    df = pd.read_json(path)

    # Ragas 0.4.x 컬럼명 확인 및 매핑
    # 예상 컬럼: user_input, reference, reference_contexts
    required_cols = []

    # 질문 컬럼 찾기
    question_col = None
    for col in ['user_input', 'question', 'query']:
        if col in df.columns:
            question_col = col
            break

    if question_col is None:
        raise ValueError(f"질문 컬럼을 찾을 수 없습니다. 컬럼: {list(df.columns)}")

    # 정답 컬럼 찾기
    answer_col = None
    for col in ['reference', 'ground_truth', 'answer', 'expected_answer']:
        if col in df.columns:
            answer_col = col
            break

    if answer_col is None:
        raise ValueError(f"정답 컬럼을 찾을 수 없습니다. 컬럼: {list(df.columns)}")

    # 컬럼명 정규화
    df = df.rename(columns={
        question_col: 'user_input',
        answer_col: 'reference'
    })

    print(f"✅ Golden Dataset 로드 완료: {len(df)}개 샘플")
    print(f"   질문 컬럼: {question_col} → user_input")
    print(f"   정답 컬럼: {answer_col} → reference")

    return df


# ============================================================
# LangGraph V1 모델 추론
# ============================================================
# ============================================================
# LangGraph V1 모델 추론
# ============================================================
def run_inference(questions: List[str], chatbot_version: str = "v3", verbose: bool = True) -> List[Dict[str, Any]]:
    """
    LangGraph V1 RAG 모델로 각 질문에 대한 답변을 생성합니다.

    Args:
        questions: 질문 리스트
        verbose: 진행 상황 출력 여부

    Returns:
        List of {answer: str, contexts: List[str]}
    """
    # chatbot_V3.py에서 LangGraph 챗봇 초기화 함수 임포트
    print(f"\n🤖 LangGraph 모델 초기화 중... (버전: {chatbot_version})")

    if chatbot_version.lower() == "v1":
        from chatbot_V1 import initialize_langgraph_chatbot
    elif chatbot_version.lower() == "v2":
        from chatbot_V2 import initialize_langgraph_chatbot
    elif chatbot_version.lower() == "v3":
        from chatbot_V3 import initialize_langgraph_chatbot
    else:
        raise ValueError(f"지원하지 않는 챗봇 버전입니다: {chatbot_version}")

    graph = initialize_langgraph_chatbot()
    print("✅ 모델 초기화 완료\n")

    results = []

    iterator = tqdm(questions, desc="추론 중") if verbose else questions

    for i, question in enumerate(iterator):
        try:
            if verbose:
                print(f"\n🔍 질문 [{i+1}]: {question}")

            # LangGraph 초기 상태 설정
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "user_query": question,
                "query_analysis": None,
                "retrieved_docs": None,
                "case_law_results": None,
                "generated_answer": None,
                "next_action": None
            }

            # 그래프 실행
            result = graph.invoke(initial_state)

            # 답변 추출
            answer = result.get("generated_answer", "")

            # 검색된 컨텍스트 추출 (LangGraph V1에서는 retrieved_docs에 Document 객체로 저장)
            contexts = []
            retrieved_docs = result.get("retrieved_docs", [])

            if retrieved_docs:
                for doc in retrieved_docs:
                    # Document 객체에서 컨텍스트 텍스트 추출
                    metadata = doc.metadata
                    source = metadata.get("source", "")
                    law_name = metadata.get("law_name", "")
                    article = metadata.get("article_no", "")
                    title = metadata.get(
                        "article_title", "") or metadata.get("title", "")
                    content = doc.page_content.strip()

                    # 컨텍스트 포맷팅 (평가에 사용할 형태로)
                    context_text = ""
                    if law_name:
                        context_text += f"[{law_name}"
                        if article:
                            context_text += f" 제{article}조"
                        context_text += "]"
                    if title:
                        context_text += f" {title}"
                    if content:
                        context_text += f"\n{content}"

                    if context_text.strip():
                        contexts.append(context_text.strip())

            # 판례 검색 결과도 컨텍스트에 추가 (있는 경우)
            case_law_results = result.get("case_law_results", [])
            if case_law_results:
                for case in case_law_results:
                    case_text = f"[판례] {case.get('title', '')}\n{case.get('content', '')}"
                    if case_text.strip():
                        contexts.append(case_text.strip())

            # 컨텍스트가 없으면 빈 리스트 대신 placeholder 사용 (fallback)
            if not contexts:
                contexts = ["(검색된 컨텍스트 없음)"]

            results.append({
                "answer": answer,
                "contexts": contexts
            })

            if verbose:
                print(f"   ✅ 컨텍스트 {len(contexts)}개 추출됨")

        except Exception as e:
            print(f"\n⚠️ 추론 실패: {question[:50]}... - {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "answer": f"[오류] {str(e)}",
                "contexts": []
            })

    return results


# ============================================================
# Ragas 평가
# ============================================================
def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    references: List[str],
    llm_model: str = "gpt-4o",
    embedding_model: Any = None
) -> Dict[str, Any]:
    """
    Ragas 메트릭으로 RAG 성능을 평가합니다.

    Args:
        questions: 질문 리스트
        answers: 생성된 답변 리스트
        contexts: 검색된 컨텍스트 리스트
        references: 정답(Ground Truth) 리스트
        llm_model: 평가에 사용할 LLM 모델

    Returns:
        평가 결과 딕셔너리
    """
    print("\n📊 Ragas 평가 시작...")

    # Ragas Dataset 생성
    eval_dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": references
    })

    # 평가용 LLM 및 Embeddings 설정
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model, temperature=0))
    eval_embeddings = LangchainEmbeddingsWrapper(
        embedding_model) if embedding_model else None

    # 메트릭 정의 (Ragas 0.4.x class-based API)
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]

    # 평가 실행
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False
        )

        print("✅ Ragas 평가 완료")
        return result

    except Exception as e:
        print(f"❌ Ragas 평가 실패: {e}")
        raise


# ============================================================
# 결과 저장
# ============================================================
def save_results(
    df: pd.DataFrame,
    ragas_result: Dict,
    output_path: str,
    chatbot_version: str = "V1"
):
    """
    평가 결과를 JSON 파일로 저장합니다.
    구조: { "summary": {metrics...}, "results": [records...] }

    Args:
        df: 원본 데이터프레임 (질문, 정답 포함)
        ragas_result: Ragas 평가 결과
        output_path: 출력 파일 경로
    """
    import json

    # Ragas 결과를 DataFrame으로 변환
    result_df = ragas_result.to_pandas()

    # 중복 컬럼 제거 (원본 df에 이미 있는 컬럼은 result_df에서 제외)
    cols_to_use = result_df.columns.difference(df.columns)

    # 원본 데이터와 결합
    final_df = pd.concat(
        [df.reset_index(drop=True),
         result_df[cols_to_use].reset_index(drop=True)],
        axis=1
    )

    # 요약 정보 생성 (평균 점수)
    # 요약 정보 생성 (평균 점수)
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": f"LangGraph {chatbot_version.upper()}",
        "metrics": {}
    }

    numeric_cols = result_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        summary["metrics"][col] = float(result_df[col].mean())

    # 최종 저장 데이터 구조
    output_data = {
        "summary": summary,
        "results": final_df.to_dict(orient='records')
    }

    # NaN 값을 None으로 변환하는 헬퍼 함수
    def replace_nan_with_none(obj):
        if isinstance(obj, float) and (obj != obj):  # Check for NaN
            return None
        if isinstance(obj, dict):
            return {k: replace_nan_with_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [replace_nan_with_none(v) for v in obj]
        return obj

    output_data = replace_nan_with_none(output_data)

    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # NaN 값을 None으로 변환하는 헬퍼 함수
    def replace_nan_with_none(obj):
        if isinstance(obj, float) and (obj != obj):  # Check for NaN
            return None
        if isinstance(obj, dict):
            return {k: replace_nan_with_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [replace_nan_with_none(v) for v in obj]
        return obj

    output_data = replace_nan_with_none(output_data)

    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장 완료: {output_path}")
    print(f"   (상단 summary 포함)")


# ============================================================
# 메인 함수
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='노동법 RAG 챗봇 평가 스크립트 (LangGraph V1)')
    parser.add_argument(
        '--golden-set',
        type=str,
        default='a_team/data/evaluation/labor_law_golden_set.json',
        help='Golden Dataset JSON 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='결과 저장 경로 (기본값: golden_set 폴더에 저장)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='평가할 샘플 수 (0이면 전체 평가)'
    )
    parser.add_argument(
        '--eval-model',
        type=str,
        default='gpt-4o',
        help='Ragas 평가에 사용할 LLM 모델'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='데이터 로드만 테스트하고 종료'
    )
    parser.add_argument(
        '--chatbot-version',
        type=str,
        default='v3',
        choices=['v1', 'v2', 'v3'],
        help='평가할 챗봇 버전 (v1, v2, v3)'
    )
    args = parser.parse_args()

    # API Key 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    print("=" * 60)
    print("🏛️  노동법 RAG 챗봇 평가 시작 (LangGraph V1)")
    print("=" * 60)

    # 1. Golden Dataset 로드
    df = load_golden_dataset(args.golden_set)

    # 샘플링
    if args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"✂️  {args.sample}개 샘플로 제한")

    # Dry run 모드
    if args.dry_run:
        print("\n🧪 Dry Run 모드: 데이터 로드만 테스트합니다.")
        print(f"\n샘플 데이터:")
        print(df.head(3).to_string())
        return

    # 2. LangGraph V1 모델 추론
    questions = df['user_input'].tolist()
    references = df['reference'].tolist()

    print(f"\n📝 {len(questions)}개 질문에 대해 추론 시작 (Chatbot {args.chatbot_version})...")
    inference_results = run_inference(
        questions, chatbot_version=args.chatbot_version)

    # 결과 추출
    answers = [r['answer'] for r in inference_results]
    contexts = [r['contexts'] for r in inference_results]

    # DataFrame에 추론 결과 추가
    df['generated_answer'] = answers
    df['retrieved_contexts'] = [str(c) for c in contexts]  # 리스트를 문자열로
    # 3. 임베딩 모델 로드 (Qwen) - Ragas 평가용
    print(f"\n🚀 평가용 임베딩 모델 로드 중 (Qwen/Qwen3-Embedding-0.6B)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Ragas 평가
    ragas_result = evaluate_with_ragas(
        questions=questions,
        answers=answers,
        contexts=contexts,
        references=references,
        llm_model=args.eval_model,
        embedding_model=embeddings
    )
    # 4. 결과 저장 (출력 전에 먼저 저장!)
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.golden_set).parent
        output_path = output_dir / f"evaluation_results_V1_{timestamp}.json"

    try:
        save_results(df, ragas_result, str(output_path),
                     chatbot_version=args.chatbot_version)
    except Exception as e:
        print(f"⚠️ 결과 저장 중 오류 발생: {e}")

    # 5. 결과 출력 (DataFrame 사용)
    print("\n" + "=" * 60)
    print("📊 평가 결과 요약 (LangGraph V1)")
    print("=" * 60)

    try:
        # Ragas 결과를 DataFrame으로 변환하여 평균 계산
        result_df = ragas_result.to_pandas()
        numeric_cols = result_df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            avg_score = result_df[col].mean()
            print(f"  • {col}: {avg_score:.4f}")

    except Exception as e:
        print(f"⚠️ 결과 출력 중 오류 발생 (데이터는 저장됨): {e}")

    print("\n" + "=" * 60)
    print("✅ 평가 완료! (LangGraph V1)")
    print("=" * 60)


if __name__ == "__main__":
    main()
