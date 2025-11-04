## Math 디렉토리 확장 계획

AI/ML 실무 관점에서 추가가 필요한 수학 노트 목록과 우선순위를 정리합니다.

---

## 현재 Math 디렉토리 구조

```
Math/
├── Calculus/
│   ├── 1. 미분과 미분법.md (완료)
│   ├── 2. 단변수 미적분.md (완료)
│   ├── 3. 다변수 미적분.md (완료)
│   ├── 4. 테일러 급수와 최적화.md (완료)
│   └── 5. 자동 미분.md (완료)
├── Linear Algebra/
│   ├── 1. 선형대수 핵심 개요.md (완료)
│   ├── 2. 벡터 공간과 기저.md (완료)
│   ├── 3. 행렬과 선형 변환.md (완료)
│   ├── 4. 고유값과 고유벡터.md (완료)
│   ├── 5. 특잇값 분해.md (완료)
│   └── 6. 슈타이니츠 교환 정리.md (완료)
├── Optimization/
│   ├── 1. 최적화 기초.md (완료)
│   ├── 2. 경사하강법과 변형.md (완료)
│   ├── 3. 2차 최적화 기법.md (완료)
│   └── 4. 볼록성과 제약 최적화.md (완료)
├── Probability/
│   ├── 1. 확률과 통계 입문.md (완료)
│   ├── 2. 확률 기초.md (완료)
│   ├── 3. 통계적 추론.md (완료)
│   └── 4. 정보 이론.md (완료)
├── Discrete/
│   ├── 1. 그래프 이론.md (완료)
│   ├── 2. 그래프 기초와 표현.md (완료)
│   ├── 3. 그래프 알고리즘.md (완료)
│   └── 4. 스펙트럴 그래프 이론과 GNN.md (완료)
└── Reference/
    └── 개념 용어집.md (완료)
```

---

## 추가 필요 항목 (우선순위별)

### 1순위(필수): 즉시 추가 권장

딥러닝 핵심 이론 이해와 실무에 직접적으로 필수적인 내용

#### 1.1 Math/Calculus/6. 벡터 미적분.md

**배경**:
- 현재 다변수 미적분이 있지만 벡터장과 미분 연산자가 부족
- Transformer attention, normalizing flow 등 이해에 필수

**주요 내용**:
- 그래디언트(Gradient), 발산(Divergence), 회전(Curl)
- 방향 도함수와 그래디언트의 관계
- 야코비안 행렬과 행렬식
- 헤시안 행렬과 2차 최적화
- 다변수 체인룰의 엄밀한 확장

**AI 연결**:
- 역전파에서 야코비안의 역할
- Attention 메커니즘의 그래디언트
- Normalizing Flow의 야코비안 행렬식
- 2차 최적화(Newton, Quasi-Newton)

**예상 분량**: 400-450 lines

**완료 기준**:
- 그래디언트·발산·회전 및 야코비안/헤시안 성질을 도표와 함께 정리
- PyTorch autograd로 야코비안/헤시안 계산 예제를 실행해 결과 해설 첨부
- 역전파·2차 최적화 연결 사례 최소 2건 기술 및 관련 노트 링크([[5. 자동 미분]] 등) 삽입
- TL;DR, 실습, 실수 주의 섹션을 템플릿에 맞춰 채우고 참고 문헌 표기

---

#### 1.2 Math/Probability/5. 베이지안 추론.md

**배경**:
- 확률 기초는 있지만 베이지안 방법론이 부재
- VAE, Bayesian Neural Network, Bayesian Optimization 이해 필수

**주요 내용**:
- 베이즈 정리와 사전/사후 확률
- 켤레 사전분포(Conjugate Prior)
- 베이지안 네트워크
- 변분 추론(Variational Inference)
- MCMC와 깁스 샘플링
- 베이지안 선형 회귀

**AI 연결**:
- VAE(Variational Autoencoder)
- Bayesian Neural Network
- Bayesian Optimization
- 불확실성 추정(Uncertainty Quantification)
- Dropout as Bayesian Approximation

**예상 분량**: 500-550 lines

**완료 기준**:
- 베이즈 정리·사전/사후·케이스 스터디를 표로 정리하고 수식 검산
- 변분 추론(ELBO) 유도를 단계별로 기술하고 PyTorch 또는 Pyro 코드로 확인
- MCMC/깁스 샘플링 절차를 의사코드→실제 코드로 연결하며 수렴 진단 포함
- VAE 혹은 Bayesian Linear Regression 예제를 실행해 로그 출력 또는 시각화 결과 캡처

---

#### 1.3 Math/Linear Algebra/7. 저랭크 근사와 행렬 분해.md

**배경**:
- SVD는 있지만 다른 분해 기법과 저랭크 응용이 부족
- 모델 압축, 효율적 학습에 직접 활용

**주요 내용**:
- QR 분해와 그람-슈미트
- LU 분해
- Cholesky 분해
- 저랭크 근사(Low-rank Approximation)
- Truncated SVD
- 행렬 완성(Matrix Completion)
- 텐서 분해 입문(Tucker, CP)

**AI 연결**:
- LoRA(Low-Rank Adaptation)
- Matrix Factorization for Recommender Systems
- 모델 압축과 경량화
- Attention의 저랭크 근사
- Tensor Decomposition for CNN

**예상 분량**: 450-500 lines

**완료 기준**:
- QR/LU/Cholesky 분해 절차를 단계별 표로 비교하고 수치 조건 논의
- Truncated SVD와 저랭크 근사 오차 경계를 식과 함께 제시
- LoRA·Matrix Factorization 사례 각각에 대해 실습 코드와 성능 비교 수치 보고
- 텐서 분해 섹션에 Tucker 또는 CP 간단 구현 및 관련 [[특잇값 분해]] 링크 추가

---

### 1.5순위(강력 추천): 최신 연구 활용

최신 딥러닝 연구와 특수 응용 분야에서 활발히 사용

#### 1.4 Math/Analysis/1. 라플라스 변환과 미분방정식.md

**배경**:
- Neural ODE, Physics-Informed Neural Networks 등 미분방정식 기반 모델 증가
- Laplace Approximation for Bayesian DL (2023-2024 활발)

**주요 내용**:
- 라플라스 변환 정의와 성질
- 역변환과 부분분수 분해
- 미분방정식 풀이
- 1계/2계 ODE 해법
- 초기값 문제(IVP)
- Neural Laplace 프레임워크

**AI 연결**:
- Neural ODE
- Neural Laplace (ICML 2022)
- Laplace Approximation for Uncertainty
- Physics-Informed Neural Networks (PINN)
- 동적 시스템 모델링

**예상 분량**: 480-530 lines

**참고 논문**:
- "Neural Laplace: Learning diverse classes of differential equations" (ICML 2022)
- "Variational Linearized Laplace Approximation for Bayesian Deep Learning" (2024)

**완료 기준**:
- 라플라스 변환·역변환 공식을 표로 정리하고 대표 ODE 풀이 2건 이상 수식 계산
- Neural Laplace·Laplace Approximation의 수학적 가정과 한계를 명문화
- SymPy 또는 torchdiffeq 기반 코드 예제를 실행해 그래프 혹은 로그 결과 첨부
- PINN·Neural ODE 등 관련 Obsidian 노트와 상호 링크 구성

---

### 2순위(유용): 특정 분야 필수

특정 AI 분야(강화학습, SVM 등)에서 필수적

#### 2.1 Math/Probability/6. 확률 과정.md

**배경**:
- 강화학습의 이론적 기초
- 시계열 모델링, 확률적 샘플링

**주요 내용**:
- 확률 과정 정의
- 마르코프 체인(이산/연속)
- 마르코프 결정 과정(MDP)
- 포아송 과정
- 브라운 운동(간단히)
- 정상 분포와 전이 확률

**AI 연결**:
- 강화학습의 MDP
- 시계열 예측(Hidden Markov Model)
- MCMC 샘플링
- Diffusion Models 입문

**예상 분량**: 420-470 lines

**완료 기준**:
- 이산/연속 마르코프 체인 정의와 성질을 비교표로 정리
- MDP와 강화학습 연결을 정책·가치함수 수식으로 서술
- NumPy 또는 PyTorch 기반 시뮬레이션 코드 실행 결과(수렴 그래프 등) 포함
- Diffusion 혹은 HMM 등 관련 노트와 상호 링크를 최소 1건 이상 추가

---

#### 2.2 Math/Optimization/5. 제약 최적화 심화.md

**배경**:
- 현재 볼록성과 제약 최적화가 있지만 KKT 등 심화 내용 부족
- SVM, Constrained RL 등 이해 필요

**주요 내용**:
- 라그랑주 승수법 복습
- KKT(Karush-Kuhn-Tucker) 조건 상세
- 쌍대 문제(Dual Problem)
- 강쌍대성/약쌍대성
- 투영 경사하강법(Projected Gradient Descent)
- 근접 연산자(Proximal Operator)
- ADMM(Alternating Direction Method of Multipliers)

**AI 연결**:
- SVM 최적화
- Constrained Reinforcement Learning
- Sparse Learning (L1 regularization)
- Fairness-constrained ML

**예상 분량**: 450-500 lines

**완료 기준**:
- KKT 조건·쌍대 문제를 표로 대비하고 증명 스케치 포함
- ADMM·Prox 연산자 알고리즘을 의사코드와 실제 코드로 각각 제시
- CVXPY 실습 결과의 최적값·라그랑주 승수 로그를 첨부
- [[Optimization/4. 볼록성과 제약 최적화]] 등 기존 노트와 상호 링크 추가

---

### 3순위(특수 목적): 도메인 특화

특정 응용 도메인(오디오, 비전, 대규모 시스템)에서 활용

#### 3.1 Math/Analysis/2. 푸리에 변환.md

**배경**:
- 오디오/신호 처리 분야
- 주파수 도메인 CNN

**주요 내용**:
- 푸리에 급수
- 연속 푸리에 변환(CFT)
- 이산 푸리에 변환(DFT)
- FFT 알고리즘
- 주파수 도메인 필터링
- 컨볼루션 정리

**AI 연결**:
- 오디오 처리 (Speech, Music)
- 주파수 도메인 CNN
- Spectral Graph Convolution
- FFT Convolution (효율적 합성곱)

**예상 분량**: 400-450 lines

**완료 기준**:
- 푸리에 급수/연속/이산 변환 공식을 표로 요약하고 컨볼루션 정리를 증명 또는 스케치
- NumPy/SciPy FFT 실습 코드와 주파수 스펙트럼 시각화 이미지 캡션 포함
- 주파수 도메인 CNN 혹은 오디오 처리 워크플로를 단계별로 연결
- 신호 처리 관련 기존 노트(있을 경우)와 상호 링크

---

#### 3.2 Math/Numerical/1. 수치 선형대수.md

**배경**:
- 대규모 행렬 연산의 수치적 안정성
- GPU 최적화, 대규모 모델 학습

**주요 내용**:
- 수치 안정성과 조건수
- 반복법(Jacobi, Gauss-Seidel, SOR)
- Krylov Subspace Methods
- Conjugate Gradient
- GMRES
- Preconditioning

**AI 연결**:
- 대규모 선형 시스템 풀이
- Implicit Layer (Deep Equilibrium Models)
- Hessian-Free Optimization
- GPU 최적화 기법

**예상 분량**: 380-430 lines

**완료 기준**:
- 조건수·수치 안정성 정의를 실례(행렬)와 함께 계산하여 표로 제시
- Jacobi/Gauss-Seidel/CG/GMRES 알고리즘 흐름도 또는 의사코드 제공
- PyTorch 또는 SciPy 구현으로 대규모 선형 시스템을 풀고 수렴 곡선 시각화
- Implicit Layer나 Hessian-Free Optimization 노트와 상호 링크 구성

---

#### 3.3 Math/Analysis/3. 측도론과 확률론 기초.md

**배경**:
- 엄밀한 확률론 기초
- 고급 ML 이론 연구

**주요 내용**:
- 시그마-대수
- 측도와 적분
- 르베스그 적분
- 확률 측도
- 기댓값의 엄밀한 정의
- 큰 수의 법칙, 중심극한정리 증명

**AI 연결**:
- ML 이론 연구
- Optimal Transport
- 수렴성 증명
- Generalization Theory

**예상 분량**: 350-400 lines

**완료 기준**:
- 시그마-대수, 측도, 르베스그 적분을 정의→예제→연습문제 순으로 정리
- 큰 수의 법칙·중심극한정리 증명 스케치를 포함하고 조건을 명확히 표기
- Optimal Transport나 일반화 이론과의 연결을 다이어그램으로 설명
- 관련 확률·최적화 노트와 상호 링크 추가

---

## 작업 계획

### Phase 0: 준비 작업 (사전 조건)

**목표**: 향후 노트 작성 중 경로·링크 충돌을 방지하고 허브 문서를 선제 갱신합니다.

1. Math/Analysis, Math/Numerical 폴더와 각 폴더의 템플릿 노트(필수 섹션 빈 틀 포함) 생성
2. README.md와 [[AI 기초 개요]]에 신규 폴더 플레이스홀더 및 예상 학습 순서를 추가
3. Obsidian 내 허브 노트에 새 항목에 대한 빈 링크를 걸어 TODO 상태를 추적

**완료 조건**: 새 폴더/파일에 대한 Git diff 확인, Obsidian 미리보기에서 링크 에러 없음, README·AI 기초 개요에 항목이 표시됨

### Phase 1: 핵심 기초 확립 (1순위 3개)

**목표**: 딥러닝 핵심 이론 이해를 위한 필수 수학 완성

1. **Math/Calculus/6. 벡터 미적분.md**
   - 작업 시간: 2-3시간
   - 의존성: Calculus/3 (다변수 미적분)
   - 영향: Foundations/7 (역전파), Operators

2. **Math/Probability/5. 베이지안 추론.md**
   - 작업 시간: 3-4시간
   - 의존성: Probability/2 (확률 기초)
   - 영향: VAE, Bayesian Optimization 노트 작성 가능

3. **Math/Linear Algebra/7. 저랭크 근사와 행렬 분해.md**
   - 작업 시간: 2.5-3.5시간
   - 의존성: Linear Algebra/5 (SVD)
   - 영향: 모델 압축 노트 작성 가능

**예상 총 작업 시간**: 7.5-10.5시간

---

### Phase 2: 최신 연구 연결 (1.5순위 1개)

**목표**: Neural ODE, Bayesian DL 등 최신 기법 이해

4. **Math/Analysis/1. 라플라스 변환과 미분방정식.md**
   - 작업 시간: 3-4시간
   - 의존성: Calculus/2 (미적분 기초)
   - 영향: Neural ODE, PINN 노트 작성 가능

**예상 총 작업 시간**: 3-4시간

---

### Phase 3: 특정 분야 심화 (2순위 2개)

**목표**: 강화학습, 제약 최적화 등 특정 분야 강화

5. **Math/Probability/6. 확률 과정.md**
   - 작업 시간: 2.5-3.5시간
   - 의존성: Probability/2
   - 영향: 강화학습 이론 강화

6. **Math/Optimization/5. 제약 최적화 심화.md**
   - 작업 시간: 2.5-3.5시간
   - 의존성: Optimization/4 (볼록성과 제약 최적화)
   - 영향: SVM, Constrained ML 노트

**예상 총 작업 시간**: 5-7시간

**참고**: 배치 크기와 최적화 동역학은 [[Foundations/3. 최적화]]에 통합 완료

---

### Phase 4: 도메인 특화 (3순위, 선택적)

**목표**: 오디오, 대규모 시스템 등 특수 도메인 (Phase 0에서 생성한 Analysis/Numerical 템플릿을 활용)

7. **Math/Analysis/2. 푸리에 변환.md** (오디오/신호 처리)
   - 작업 시간: 2.5-3시간

8. **Math/Numerical/1. 수치 선형대수.md** (대규모 시스템)
   - 작업 시간: 2-3시간

9. **Math/Analysis/3. 측도론과 확률론 기초.md** (이론 연구)
   - 작업 시간: 2-3시간

**예상 총 작업 시간**: 6.5-9시간

---

## 전체 타임라인 요약

| Phase | 항목 수 | 우선순위 | 예상 시간 | 누적 시간 |
|-------|---------|----------|-----------|-----------|
| Phase 1 | 3개 | 필수 | 7.5-10.5h | 7.5-10.5h |
| Phase 2 | 1개 | 강력 추천 | 3-4h | 10.5-14.5h |
| Phase 3 | 2개 | 유용 | 5-7h | 15.5-21.5h |
| Phase 4 | 3개 | 특수 목적 | 6.5-9h | 22-30.5h |

**전체 완료 예상 시간**: 22-30.5시간

**참고**: 배치 크기 관련 내용은 Foundations/3. 최적화.md에 이미 추가 완료 (2024)

---

## 완료 정의 (Definition of Done)

- PyTorch Tensor 템플릿 6개 섹션(TL;DR, 언제 쓰나, 주요 API, 실습 예제, 실수 주의, 관련 노트)을 모두 채워 일관성을 유지
- 실습 예제 코드를 최신 PyTorch 기반으로 실행하고, 출력 로그 혹은 그래프를 노트에 캡션과 함께 삽입
- README.md, [[AI 기초 개요]], 관련 허브 노트에 새 링크를 추가하고 Obsidian 미리보기에서 깨진 링크가 없음을 확인
- "실수 주의" 섹션에 최소 두 가지 오류 패턴과 대응 전략을 기록하며, 필요시 [[Math/Reference/개념 용어집]]에 용어를 추가
- 코드·수식 검증을 위해 간단한 테스트나 계산을 진행하고 결과를 노트에 요약(예: autograd 값 비교, 수치 안정성 체크)

---

## 각 항목별 상세 개요

### 1. Math/Calculus/6. 벡터 미적분.md

```markdown
## TL;DR
- 벡터장에 대한 미분 연산자(그래디언트, 발산, 회전)
- 야코비안과 헤시안 행렬의 역할과 계산
- 다변수 체인룰의 엄밀한 확장
- 역전파와 2차 최적화에 필수적

## 핵심 개념
### 그래디언트 (Gradient)
### 방향 도함수
### 야코비안 행렬
### 헤시안 행렬
### 발산(Divergence)과 회전(Curl)
### 다변수 체인룰

## 실습 예제
- PyTorch autograd로 야코비안 계산
- 헤시안 기반 2차 최적화
- Attention 메커니즘의 그래디언트 추적

## AI 연결
- 역전파의 수학적 기초
- Newton법과 Quasi-Newton법
- Normalizing Flow의 야코비안 행렬식
```

**선수 지식**: Calculus/3 (다변수 미적분)
**후속 노트**: Foundations/7 (역전파), Optimization/3 (2차 최적화)

---

### 2. Math/Probability/5. 베이지안 추론.md

```markdown
## TL;DR
- 베이즈 정리와 사전/사후 확률의 관계
- 변분 추론으로 사후 분포 근사
- MCMC와 깁스 샘플링
- VAE, Bayesian Neural Network의 이론적 기초

## 핵심 개념
### 베이즈 정리
### 사전/사후 확률과 우도
### 켤레 사전분포
### 베이지안 추론의 계산적 어려움
### 변분 추론 (Variational Inference)
### MCMC (Markov Chain Monte Carlo)
### 깁스 샘플링

## 실습 예제
- PyMC3로 베이지안 선형 회귀
- PyTorch로 간단한 VAE 구현
- Laplace 라이브러리로 Bayesian Neural Network

## AI 연결
- VAE (Variational Autoencoder)
- Bayesian Optimization
- Uncertainty Quantification
- Dropout as Bayesian Approximation
```

**선수 지식**: Probability/2 (확률 기초)
**후속 노트**: VAE 노트, Bayesian Neural Network 노트 (Architectures)

---

### 3. Math/Linear Algebra/7. 저랭크 근사와 행렬 분해.md

```markdown
## TL;DR
- QR, LU, Cholesky 등 다양한 행렬 분해 기법
- 저랭크 근사로 데이터와 모델 압축
- LoRA, Matrix Factorization의 수학적 기초

## 핵심 개념
### QR 분해와 그람-슈미트
### LU 분해
### Cholesky 분해
### 저랭크 근사 (Low-rank Approximation)
### Truncated SVD
### 행렬 완성 (Matrix Completion)
### 텐서 분해 입문 (Tucker, CP)

## 실습 예제
- NumPy/PyTorch로 각종 분해 구현
- LoRA 구조 이해와 적용
- Recommender System의 Matrix Factorization

## AI 연결
- LoRA (Low-Rank Adaptation) for LLM
- Matrix Factorization for RecSys
- 모델 압축과 경량화
- Attention의 저랭크 근사
```

**선수 지식**: Linear Algebra/5 (SVD)
**후속 노트**: 모델 압축 노트, LoRA 노트 (Architectures)

---

### 4. Math/Analysis/1. 라플라스 변환과 미분방정식.md

```markdown
## TL;DR
- 라플라스 변환으로 미분방정식을 대수 방정식으로 변환
- Neural Laplace로 동적 시스템 모델링 (O(1) 복잡도)
- Laplace Approximation으로 신경망 불확실성 추정

## 핵심 개념
### 라플라스 변환 정의와 성질
### 역변환과 부분분수 분해
### 1계/2계 ODE 풀이
### 초기값 문제 (IVP)
### Neural Laplace 프레임워크
### Laplace Approximation for Bayesian DL

## 실습 예제
- SymPy로 라플라스 변환 계산
- PyTorch로 Neural ODE 구현
- laplace 라이브러리로 불확실성 추정

## AI 연결
- Neural ODE
- Neural Laplace (ICML 2022)
- Physics-Informed Neural Networks (PINN)
- Bayesian Deep Learning
```

**선수 지식**: Calculus/2 (미적분 기초)
**후속 노트**: Neural ODE 노트, PINN 노트 (Architectures)
**참고 논문**:
- Neural Laplace (ICML 2022)
- Variational Linearized Laplace Approximation (2024)

---

### 5. Math/Probability/6. 확률 과정.md

```markdown
## TL;DR
- 시간에 따라 변화하는 확률 시스템
- 마르코프 체인과 MDP: 강화학습의 기초
- MCMC 샘플링의 이론적 배경

## 핵심 개념
### 확률 과정 정의
### 마르코프 체인 (이산/연속)
### 전이 확률과 정상 분포
### 마르코프 결정 과정 (MDP)
### 포아송 과정
### 브라운 운동 (간단히)

## 실습 예제
- NumPy로 마르코프 체인 시뮬레이션
- OpenAI Gym으로 MDP 이해
- MCMC 샘플링 구현

## AI 연결
- 강화학습의 MDP
- MCMC (Metropolis-Hastings, Gibbs)
- Hidden Markov Model
- Diffusion Models 입문
```

**선수 지식**: Probability/2 (확률 기초)
**후속 노트**: 강화학습 노트 (Foundations 또는 별도)

---

### 6. Math/Optimization/5. 제약 최적화 심화.md

```markdown
## TL;DR
- KKT 조건으로 제약 최적화 문제 해결
- 쌍대 문제와 쌍대성 갭
- 근접 연산자와 ADMM으로 복잡한 제약 처리

## 핵심 개념
### 라그랑주 승수법 복습
### KKT (Karush-Kuhn-Tucker) 조건
### 쌍대 문제 (Primal-Dual)
### 강쌍대성과 약쌍대성
### 투영 경사하강법
### 근접 연산자 (Proximal Operator)
### ADMM

## 실습 예제
- CVXPY로 제약 최적화 문제 풀기
- PyTorch로 투영 경사하강법 구현
- L1 정규화와 근접 연산자

## AI 연결
- SVM 최적화
- Sparse Learning
- Constrained Reinforcement Learning
- Fairness-constrained ML
```

**선수 지식**: Optimization/4 (볼록성과 제약 최적화)
**후속 노트**: SVM 노트, Constrained ML 노트

---

### 7. Math/Analysis/2. 푸리에 변환.md

```markdown
## TL;DR
- 시간 도메인 ↔ 주파수 도메인 변환
- FFT로 효율적인 합성곱 연산
- 오디오, 신호 처리 분야의 기초

## 핵심 개념
### 푸리에 급수
### 연속/이산 푸리에 변환
### FFT 알고리즘
### 주파수 도메인 필터링
### 컨볼루션 정리

## 실습 예제
- NumPy/SciPy로 FFT 계산
- 오디오 신호 분석
- 주파수 도메인 CNN

## AI 연결
- Speech/Music Processing
- Spectral Graph Convolution
- FFT Convolution
```

**선수 지식**: Calculus/2
**후속 노트**: 오디오 처리 노트

---

### 8. Math/Numerical/1. 수치 선형대수.md

```markdown
## TL;DR
- 대규모 행렬 연산의 수치적 안정성
- 반복법으로 효율적인 선형 시스템 풀이
- GPU 최적화와 대규모 모델 학습

## 핵심 개념
### 수치 안정성과 조건수
### 반복법 (Jacobi, Gauss-Seidel)
### Conjugate Gradient
### GMRES
### Preconditioning

## 실습 예제
- SciPy로 대규모 선형 시스템 풀이
- PyTorch로 Conjugate Gradient 구현

## AI 연결
- Implicit Layer (DEQ)
- Hessian-Free Optimization
- GPU 최적화
```

**선수 지식**: Linear Algebra/3
**후속 노트**: 대규모 최적화 노트

---

### 9. Math/Analysis/3. 측도론과 확률론 기초.md

```markdown
## TL;DR
- 엄밀한 확률론의 기초
- ML 이론 연구를 위한 수학적 토대
- Optimal Transport의 기초

## 핵심 개념
### 시그마-대수와 측도
### 르베스그 적분
### 확률 측도
### 기댓값의 엄밀한 정의
### 큰 수의 법칙, 중심극한정리 증명

## 실습 예제
- 측도론적 관점에서 확률 계산

## AI 연결
- Optimal Transport
- Generalization Theory
- ML 이론 연구
```

**선수 지식**: Calculus/2, Probability/2
**후속 노트**: Optimal Transport 노트

---

## 디렉토리 구조 변화

### 추가될 폴더

```
Math/
├── Analysis/          # 새로 생성
│   ├── 1. 라플라스 변환과 미분방정식.md
│   ├── 2. 푸리에 변환.md
│   └── 3. 측도론과 확률론 기초.md
└── Numerical/         # 새로 생성
    └── 1. 수치 선형대수.md
```

### 확장될 폴더

```
Math/
├── Calculus/
│   └── 6. 벡터 미적분.md          # 추가
├── Linear Algebra/
│   └── 7. 저랭크 근사와 행렬 분해.md  # 추가
├── Optimization/
│   └── 5. 제약 최적화 심화.md      # 추가
└── Probability/
    ├── 5. 베이지안 추론.md         # 추가
    └── 6. 확률 과정.md             # 추가
```

---

## 작업 체크리스트

### Phase 0: 준비 작업

- [ ] Math/Analysis/1~3 템플릿 노트 생성(TL;DR~관련 노트 섹션 빈 틀 포함)
- [ ] Math/Numerical/1 템플릿 노트 생성
- [ ] README.md와 [[AI 기초 개요]]에 신규 폴더 및 예상 로드맵 플레이스홀더 추가
- [ ] Foundations·Operators 허브 노트에 빈 링크 추가 후 Obsidian 미리보기에서 링크 상태 확인

### Phase 1: 핵심 기초 확립

- [ ] Math/Calculus/6. 벡터 미적분.md
  - [ ] 그래디언트, 발산, 회전 정의와 성질
  - [ ] 야코비안, 헤시안 행렬
  - [ ] 다변수 체인룰
  - [ ] PyTorch autograd 예제
  - [ ] 역전파 연결
  - [ ] 실습 코드 실행 로그·시각화 캡처 및 재현 절차 기록
  - [ ] README·[[AI 기초 개요]]·관련 노트에 링크 업데이트 및 검증
  - [ ] "실수 주의" 섹션에 최소 2개 오류 패턴 추가

- [ ] Math/Probability/5. 베이지안 추론.md
  - [ ] 베이즈 정리와 사전/사후 확률
  - [ ] 변분 추론 (ELBO 유도)
  - [ ] MCMC, 깁스 샘플링
  - [ ] VAE 구현 예제
  - [ ] Laplace 라이브러리 예제
  - [ ] PyMC/Pyro 실행 결과 로그 수집 및 노트에 요약
  - [ ] README·[[AI 기초 개요]]·관련 노트 링크 업데이트
  - [ ] 실습 노트북/스크립트 재현 방법 정리

- [ ] Math/Linear Algebra/7. 저랭크 근사와 행렬 분해.md
  - [ ] QR, LU, Cholesky 분해
  - [ ] 저랭크 근사 이론
  - [ ] Truncated SVD
  - [ ] LoRA 구조 설명
  - [ ] Matrix Factorization 예제
  - [ ] 저랭크 실험 결과(압축률·성능) 표로 정리
  - [ ] README·[[AI 기초 개요]]·관련 노트 링크 업데이트
  - [ ] 관련 코드/노트북 실행 로그 검증

### Phase 2: 최신 연구 연결

- [ ] Math/Analysis/1. 라플라스 변환과 미분방정식.md
  - [ ] 라플라스 변환 정의와 성질
  - [ ] ODE 풀이
  - [ ] Neural Laplace 프레임워크
  - [ ] Laplace Approximation
  - [ ] Neural ODE 구현 예제
  - [ ] SymPy 또는 torchdiffeq 실행 결과 삽입 및 링크 검증
  - [ ] README·[[AI 기초 개요]]·관련 노트 링크 업데이트
  - [ ] 실수 주의 섹션에 수치적 함정 정리

### Phase 3: 특정 분야 심화

- [ ] Math/Probability/6. 확률 과정.md
  - [ ] 마르코프 체인
  - [ ] MDP 정의
  - [ ] 전이 확률과 정상 분포
  - [ ] 강화학습 연결
  - [ ] 시뮬레이션 코드 실행 및 수렴 그래프 삽입
  - [ ] README·[[AI 기초 개요]]·강화학습 노트 링크 업데이트

- [ ] Math/Optimization/5. 제약 최적화 심화.md
  - [ ] KKT 조건 상세
  - [ ] 쌍대 문제
  - [ ] 근접 연산자, ADMM
  - [ ] SVM 예제
  - [ ] CVXPY/Projected GD 실행 결과와 파라미터 값을 표로 정리
  - [ ] README·[[AI 기초 개요]]·관련 최적화 노트 링크 업데이트

### Phase 4: 도메인 특화 (선택적)

- [ ] Math/Analysis/2. 푸리에 변환.md
  - [ ] FFT/DFT 실습 코드 실행 및 주파수 스펙트럼 시각화
  - [ ] 신호 처리 관련 노트와 상호 링크 추가
  - [ ] README·[[AI 기초 개요]] 링크 업데이트

- [ ] Math/Numerical/1. 수치 선형대수.md
  - [ ] 반복법/CG/GMRES 코드 실행과 수렴 그래프 삽입
  - [ ] 조건수·안정성 비교 표 작성
  - [ ] README·[[AI 기초 개요]] 링크 업데이트

- [ ] Math/Analysis/3. 측도론과 확률론 기초.md
  - [ ] 측도·적분 예제 계산 및 증명 스케치 정리
  - [ ] Optimal Transport·확률 노트와 상호 링크 추가
  - [ ] README·[[AI 기초 개요]] 링크 업데이트

---

## 관련 노트 업데이트 필요

새 Math 노트 추가 시 다음 노트들도 업데이트 필요:

1. **README.md** - Math 섹션에 새 파일 추가
2. **AI 기초 개요.md** - Math 학습 로드맵 업데이트
3. **Foundations/** 파일들 - Math 노트 링크 추가
4. **Operators/** 파일들 - 관련 수학 개념 링크

---

## 참고 자료

### 핵심 교재
- Goodfellow et al., *Deep Learning* (2016) - Math for DL
- Bishop, *Pattern Recognition and Machine Learning* (2006)
- Boyd & Vandenberghe, *Convex Optimization* (2004)

### 최신 논문 (라플라스 관련)
- Holt et al., "Neural Laplace: Learning diverse classes of differential equations" (ICML 2022)
- Daxberger et al., "Laplace Redux - Effortless Bayesian Deep Learning" (NeurIPS 2021)
- Ortega et al., "Variational Linearized Laplace Approximation for Bayesian Deep Learning" (ICML 2024)

### 온라인 강의
- MIT 18.06: Linear Algebra (Gilbert Strang)
- Stanford CS229: Machine Learning (Math Review)
- Stanford EE364a: Convex Optimization

---

## 다음 단계

이 계획서 작성 완료 후:

1. **우선순위 확정**: Phase 1부터 시작할지 사용자 확인
2. **첫 번째 파일 작성**: Math/Calculus/6. 벡터 미적분.md
3. **점진적 확장**: 한 번에 하나씩 완성하며 검토
4. **연결 강화**: 완성된 Math 노트를 Foundations, Operators와 링크

작업을 시작하시겠습니까?
