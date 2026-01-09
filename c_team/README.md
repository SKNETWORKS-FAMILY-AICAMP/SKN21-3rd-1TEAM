

# C-TEAM 형사법 RAG Chatbot & Data Processing

본 문서는 **C-TEAM 형사법 RAG 챗봇**의 전체 아키텍처, 데이터 수집·전처리 파이프라인, 체인 구성, 성능 검증 구조를 기술합니다. 

---

## 1. 시스템 아키텍처 (System Architecture)

### 1.1 개요

본 프로젝트는 **형사법(형법·형사소송법·특별형법)** 도메인에 특화된 **Retrieval-Augmented Generation (RAG)** 챗봇입니다.

* LangChain 기반 체인 구조
* Vector Retrieval + LLM Answer Generation
* 판례/법령 기반 근거 제시(Evidence-based Answer)

핵심 목표는 **형사법 질의에 대해 관련 법 조문을 정확히 검색하고, 출처 기반 답변을 생성**하는 것입니다.

---

### 1.2 계층 구조 (Layers)

#### 1) Configuration

* `.env` 기반 환경 변수 관리
* OpenAI API Key 및 모델 설정

#### 2) Infrastructure

* **LLM**: OpenAI Chat Model
* **Vector Store**: Qdrant 임베딩 기반 문서 검색
* **Embedding**: 문서 벡터화

#### 3) Logic (Core Pipeline)

* Prompt 구성
* Retrieval → Context 주입 → Answer Generation

#### 4) Execution

* 체인 실행
* 성능 검증 및 평가

---

### 1.3 전체 데이터 흐름 (Workflow)

```text
[ User Query ]
     │
     ▼
[ Retriever ]  ← 형사법 문서(Vector DB)
     │
     ▼
[ Context Injection ]
     │
     ▼
[ LLM Answer Generation ]
     │
     ▼
[ Final Answer + Evidence ]
```

---

## 2. 체인 구조 (LangChain Pipeline)

### 2.1 chaining.py

`chaining.py`는 형사법 RAG 챗봇의 **핵심 체인 로직**을 담당합니다.

#### 주요 특징

* PromptTemplate + LLM + OutputParser 구조
* Retriever를 통한 문서 검색
* 검색된 문서를 Context로 주입

#### 체인 흐름

1. 사용자 질문 입력
2. Retriever가 관련 형사법 문서 검색
3. Prompt에 문서 + 질문 삽입
4. LLM이 최종 답변 생성

---

## 3. 데이터 수집 (Data Crawling)

### 3.1 crawling.ipynb

형사법 RAG의 기반 데이터는 **웹 크롤링**을 통해 수집됩니다.

#### 수집 대상

* 형사소송법

#### 기술 스택

* Selenium
* WebDriver
* HTML 파싱

#### 주요 처리 내용

* 페이지 구조 분석
* 본문 텍스트 추출
* 불필요한 태그 제거
* JSON/텍스트 형태로 저장
* 평가지표 신뢰도를 높이기 위해 2025 경찰 형사법 시험 기반 커스텀 데이터셋 구축

---

## 4. 데이터 전처리 & 구조화

### 4.1 전처리 목적

* 형사법 문서를 **LLM 친화적 형태**로 변환
* 검색 정확도 향상
* 문맥 보존

### 4.2 주요 전처리 전략

* 조문 단위 텍스트 정리
* 판례는 사건 단위로 묶어 구조화
* 불필요한 공백, 특수문자 제거

---

## 5. 성능 검증 (Performance Verification)

### 5.1 performance_verification.ipynb

본 노트북은 **RAG 체인의 성능 검증**을 담당합니다.

#### 주요 기능

* chaining.py에서 정의한 체인 로드
* 테스트 질문 세트 실행
* 응답 결과 확인

#### 평가 관점

* 답변의 형사법 적합성
* 근거 문서 활용 여부
* 질문-답변 일관성

---

## 6. 디렉토리 구조

```text
C-TEAM/
├── chaining.py                  # RAG 체인 핵심 로직
├── crawling.ipynb               # 형사법 데이터 크롤링
├── performance_verification.ipynb # 성능 검증
├── data/                        #  데이터
├── show.ipynb
├── split_data.ipynb
├── stores.py
├── vectorization.ipynb
├── verification_datas.py
└── README.md
```

---

## 7. 트러블 슈팅 (Troubleshooting & Challenges)

프로젝트 진행 과정에서 마주친 주요 기술적 난관과 이를 해결하기 위한 시도들입니다. 

### 7.1. Context Length 문제 

- **문제점** : 초기에는 장 단위로 텍스트를 분할해 검색된 컨텍스트가 길어 검색 결과 3개 (k=3)의 개수가 LLM의 최대 출력 토큰 제한을 초과해 성능 평가 점수가 좋지 않았습니다.
- **해결** : 장, 절, 조 단위로 텍스트를 분할하고, 컨텍스트 검색 및 길이 제한을 뒀으며 검색 결과 개수를 조정 (k=2)했으나 완벽하게 해결되지는 않았습니다.

### 7.2. 프롬프트 최적화 및 평가 지표 불일치 문제 

- **문제점**: 모델이 제공된 컨텍스트를 무시하고 학습된 내부 지식으로 답변해 성능 평가 점수가 좋지 않았습니다. AI 답변 내용 자체는 법률적으로 정확했으나 모델이 검색된 문구를 직접 인용하지 않아 근거를 찾지 못했기 때문입니다. 
- **시도**: 인용 가이드 라인을 추가해 반드시 제공된 문맥 내의 조문 번호와 핵심 법률 용어를 직접 인용하도록 프롬프트를 보강했습니다. 단순히 구어체 질문만으로 검색하지 않고, 질문 앞에 "형법 조문 원문 구성요건" 등의 법률적 키워드를 결합해 리트리버가 실제 정답과 일치하는 조문을 찾아올 확률을 높였습니다. 

---

## 8. 향후 개선 방향

* Reranker 도입으로 유사 판례 혼동 감소
* 평가 지표 정량화 (RAGAS 등)

---

