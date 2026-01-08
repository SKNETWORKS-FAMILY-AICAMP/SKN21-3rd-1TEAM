# A-TEAM Chatbot & Data Processing

본 문서는 A-TEAM의 노동 법률 RAG 챗봇 아키텍처와 데이터 전처리 로직에 대해 기술합니다.

## 1. 시스템 아키텍처 (System Architecture)

### 1.1. 개요

`chatbot_graph_V8.py`는 **LangGraph**를 기반으로 한 상태 기반(Stateful) 에이전트 워크플로우를 구현합니다.

### 1.2. 계층 구조 (Layers)

- **Configuration**: `Config` 클래스에서 모든 설정(모델명, 임계값, 프롬프트 등)을 중앙 관리
- **Infrastructure**: 외부 리소스 연결 관리
  - **Vector Store**: Qdrant (HyDE / Semantic Search)
  - **Keyword Search**: BM25 (Lexical Search)
  - **Reranker**: Jina Reranker (Multilingual)
- **Logic**: 주요 노드별 비즈니스 로직
  - `Analyze`: 질문 의도 및 유형 분석
  - `Search`: HyDE + Hybrid Search + Reranking + Boosting
  - `Generate`: Context 기반 답변 생성 (Chain of Thought)
  - `Evaluate`: 답변 품질 평가
- **Execution**: 그래프 컴파일 및 실행

### 1.3. 데이터 흐름 (Workflow)

```text
[ User Input ]
      │
      ▼
[ 🧠 Analyze Query ] ──(정보가 더 필요해??)──▶ [ 🗣️ 정보 요청 ]
      │
      │ (Ready)
      ▼
[ 🔀 Query Expansion ]
      │
      ├──▶  Vector Search (Qdrant)
      ├──▶  Keyword Search (BM25)
      ├──▶  HyDE
      ▼
[ 📉 Jina Reranking + Filtering ]
      │
      ▼
[ ✍️ Generate Answer (CoT) ]
      │
      ▼
[ ⚖️ Evaluate Quality ] ──(Fail)──▶ (Retry Search)
      │
      │ (Pass)
      ▼
[ ✅ Final Answer ]
```

![alt text](./data/images/image.png)

1. **Analyze**: 사용자 질문을 분석하여 `intent_type`(법령조회/판례검색 등)과 `category`(노동/민사 등)를 파악합니다.
2. **Clarify**: 질문이 너무 모호한 경우 명확화를 위한 역질문을 생성합니다.
3. **Search**:
   - **Query Expansion**: LLM을 사용하여 **HyDE(가상 답변)** 및 **Keywords**를 생성합니다.
   - **Hybrid Retrieval**: Vector Search(Qdrant)와 Keyword Search(BM25) 결과를 결합합니다.
   - **Reranking**: Jina Reranker를 사용해 의미적 연관성 순으로 재정렬합니다.
   - **Filtering**: 관련 법령(`related_laws`)에 가중치를 부여하고, 유사도(`0.2`) 미만 문서를 필터링합니다.
4. **Generate**: 검색된 문서를 바탕으로 엄격한 증거 기반(Evidence-based) 답변을 생성합니다. Hallucination 방지를 위해 출처를 명시합니다.
5. **Evaluate**: 생성된 답변의 정확성과 문서 인용 여부를 평가하여, 기준 미달 시 재검색을 수행합니다.

---

## 2. 데이터 전처리 (Data Preprocessing)

### 2.1. 법령 데이터 (`preprocesser_법령.py`)

- **대상 파일**: `rd_노동법.json`, `rd_민사법.json`, `rd_형사법.json`
- **전처리 로직**:
  - **HTML 정제**: 불필요한 태그 제거 및 `<개정 2021. 1. 5.>` 형태를 `[개정 2021.1.5]`로 간소화.
  - **헤더 정규화**: `부      칙` -> `부칙`, `별       표` -> `별표` 등의 공백 정규화.
- **청킹 및 구조화**:
  - **조문 (Article)**: 각 조문을 기본 청크 단위로 합니다.
    - 내용이 **500자**를 초과할 경우, **100자 Overlap**을 적용하여 분할합니다.
  - **별표 (Table)**: 텍스트 길이가 **300자** 미만인 경우 해당 조문에 **병합**하여 문맥을 보존하고, 그 이상인 경우 독립된 청크로 분리합니다.
  - **부칙**: 최신 부칙만 유지하여 처리합니다.

### 2.2. 법령 외 데이터 (`preprocesser_법령외.py`)

- **대상 파일**: 주요판정사례, 행정해석, 판정선례, 고용노동부 Q&A
- **텍스트 구조화 (Context Reconstruction)**:
  - 단순 텍스트 나열이 아닌, "질문-답변" 구조를 하나로 묶어 문맥을 유지합니다.
  - **주요판정사례**: `[제목] \n 판정사항: ... \n 판정요지: ...`
  - **행정해석/Q&A**: `[제목] \n 질의: ... \n 답변: ...`
  - **판정선례**: `[제목] \n [질의] ... \n [회신] ... \n [관련법령] ...`
- **벡터화 시 후처리 (`vectorize_to_qdrant_결정선례.py`)**:
  - 전처리된 텍스트가 Qdrant에 저장되기 전, 너무 긴 문서(예: 판정선례)는 **800자 Chunk / 100자 Overlap**으로 다시 분할되어 저장됩니다.

---

## 3. 트러블슈팅 (Troubleshooting & Challenges)

프로젝트 진행 과정에서 마주친 주요 기술적 난관과 이를 해결하기 위한 시도들입니다.

### 3.1. 청킹(Chunking) 시 법령 컨텍스트 소실 문제

- **문제점**: 초기에는 단순 조항 단위로 텍스트를 분할했습니다. 이로 인해 청크만 보았을 때 해당 조항이 "근로기준법"인지 "노동조합법"인지 식별하기 어려워, 검색 정확도가 떨어지는 현상이 발생했습니다.
- **해결**: 각 청크의 맨 앞에 **`[법령명] 제N조(제목)` 형태의 헤더를 강제로 추가**했습니다. 이를 통해 Embedding 모델이 해당 텍스트가 어떤 법령에 속하는지 명확히 인지하게 되어 검색 성능이 크게 향상되었습니다.

### 3.2. 평가 데이터셋(Golden Dataset)의 품질 이슈

- **문제점**: 초기 생성된 평가용 질문(Question)들이 사람이 묻는 자연스러운 질문이 아니라, 법조문의 문장을 그대로 비틀거나 지나치게 예외적인 상황만을 다루는 경우가 많았습니다. 이로 인해 Ragas 등의 평가 점수(Faithfulness, Relevance 등)를 신뢰할 수 없었습니다.
- **시도**: "일반인이 질문하는 듯한 구어체"와 "구체적인 위반 상황"을 프롬프트에 명시하여 데이터셋을 재생성했습니다.

### 3.3. 유사 조항 간의 의미적 혼동

- **문제점**: 품질이 개선된 데이터셋으로 실험했을 때, 모델이 법령의 내용은 유지하지만 **의미론적으로 매우 유사한 다른 조항**을 정답으로 가져오는 문제가 발생했습니다. (예: 해고 예고와 해고 사유 제한은 다르지만, '해고'라는 맥락에서 벡터 유사도가 높게 잡힘)
- **분석**: 법률 데이터 특성상, 용어가 겹치면 문맥이 달라도 벡터값이 가깝게 위치하는 한계가 있었습니다. 이를 해결하기 위해 Reranker를 도입했으나 완벽하게 해결되지는 않았습니다.

### 3.4. 검색 성능(Retrieval Performance)의 한계

- **난관**: HyDE, Query Expansion, Hybrid Search, Reranking 등 다양한 SOTA 기법들을 적용했음에도 불구하고, "어떻게 하면 정확한 문서를 **100%** 찾아올 것인가"에 대한 명쾌한 해결책(Silver Bullet)을 찾지 못했습니다.

---

## 4. 평가 결과 (Evaluation Results)

각 버전별 Ragas 평가 결과입니다. (Context Precision은 `llm_context_precision_without_reference` 메트릭을 사용했습니다.)

| Version | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Timestamp           |
| :------ | :----------- | :--------------- | :---------------- | :------------- | :------------------ |
| **V1**  | 0.6482       | 0.4783           | 0.7917            | 0.3667         | 2026-01-08 14:55:22 |
| **V2**  | 0.4957       | 0.5623           | 0.8462            | 0.6346         | 2026-01-07 22:45:58 |
| **V3**  | 0.5381       | 0.5602           | 0.7804            | 0.6603         | 2026-01-07 23:45:11 |
| **V3**  | 0.6522       | 0.6646           | 0.6739            | 0.4926         | 2026-01-08 09:41:41 |
| **V4**  | 0.6166       | 0.6608           | 0.6693            | 0.4708         | 2026-01-08 10:46:37 |
| **V4**  | 0.6398       | 0.7090           | 0.7506            | 0.6000         | 2026-01-08 10:59:11 |
| **V5**  | 0.5467       | 0.6319           | 0.8533            | 0.5667         | 2026-01-08 11:41:26 |
| **V7**  | 0.5894       | 0.7110           | 0.7211            | 0.5333         | 2026-01-08 12:14:14 |
| **V7**  | 0.5792       | 0.7090           | 0.7634            | 0.5333         | 2026-01-08 12:28:11 |
| **V8**  | 0.5873       | 0.5121           | 0.8012            | 0.4333         | 2026-01-08 14:13:39 |
| **V8**  | 0.5897       | 0.5804           | 0.6233            | 0.4000         | 2026-01-08 14:18:55 |

## 4. 디렉토리 구조

### 4.1. 트리 구조

<img src="SKN21-3rd-1Team-A_directory.png" width="60%" alt="A-Team Directory Structure">

### 4.2. 디렉토리별 설명

#### 📁 data/

- **raw/**: 크롤링한 원본 JSON 및 PDF 파일
  - 법령 데이터 (노동법, 민사법, 형사법)
  - 법령외 데이터 (결정선례, QA, 판정사례, 행정해석)
- **processed/**: 청킹 및 전처리된 데이터
  - 벡터 DB 저장을 위해 가공된 데이터
- **evaluation/**: 평가 관련 데이터
  - Golden dataset (V1, V2_10, V2_20)
  - 평가 결과 (baseline, V1~V8)

#### 📁 scripts/

- **architectures/**: 챗봇 구현 버전들
  - [chatbot_baseline.py](scripts/architectures/chatbot_baseline.py): 기본 구현
  - [chatbot_chain_V2.py](scripts/architectures/chatbot_chain_V2.py) ~ [chatbot_chain_V3.py](scripts/architectures/chatbot_chain_V3.py): LangChain 기반
  - [chatbot_graph_V8_FINAL.py](scripts/architectures/chatbot_graph_V8_FINAL.py): LangGraph 기반 최신 버전 ⭐
- **crawlers/**: 데이터 수집(크롤링) 스크립트
- **data_preprocessing/**: 데이터 전처리
  - [preprocesser\_법령.py](scripts/data_preprocessing/preprocesser_법령.py), [preprocesser\_법령외.py](scripts/data_preprocessing/preprocesser_법령외.py): 전처리 스크립트
  - [vectorizer\_법령.py](scripts/data_preprocessing/vectorizer_법령.py), [vectorizer\_법령외.py](scripts/data_preprocessing/vectorizer_법령외.py): 벡터화 스크립트
- **평가 및 생성 스크립트**:
  - [evaluate_rag_baseline.py](scripts/evaluate_rag_baseline.py) ~ [evaluate_rag_Vfinal.py](scripts/evaluate_rag_Vfinal.py): RAG 평가
  - [generate_evaldata_V2.py](scripts/generate_evaldata_V2.py): 평가 데이터셋 생성

#### 통계

- 총 파일 수: 약 60개
- 총 데이터 크기: ~200 MB (raw + processed)
- 챗봇 버전: 12개 (baseline + chain 2개 + graph 8개)
- 평가 결과: 14개
