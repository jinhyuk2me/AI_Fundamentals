# AI Basics - 딥러닝을 위한 수학·프로그래밍·ML 기초

> 교과서 수준의 체계성과 엄밀함으로 구성된 AI/딥러닝 학습 노트 모음

## 📚 프로젝트 소개

이 저장소는 **Obsidian Vault**로 관리되는 AI/딥러닝 기초 학습 노트입니다. 수학적 엄밀함과 실무 활용의 균형을 맞춰 대학 교과서 수준으로 작성되었으며, 단계별 학습 경로를 통해 체계적으로 기초를 다질 수 있습니다.

### 핵심 특징

- ✅ **교과서 수준의 체계성**: 정의 → 정리 → 예제 → 실습 → 연습문제 순서로 구성
- ✅ **수학과 코드의 균형**: 이론적 엄밀함과 Python/PyTorch 실습 예제를 함께 제공
- ✅ **연결된 학습 경로**: Obsidian 양방향 링크로 관련 개념들을 체계적으로 연결
- ✅ **실무 중심 설명**: 딥러닝/머신러닝 실무와의 연결점을 명시적으로 제시
- ✅ **한국어 작성**: 모든 내용이 한국어로 작성되어 있으며, 영문 용어는 병기

## 🗂️ 저장소 구조

```
ai_basics/
├── Math/                      # 수학 기초
│   ├── Linear Algebra/        # 선형대수
│   ├── Calculus/              # 미적분
│   ├── Probability/           # 확률과 통계
│   ├── Optimization/          # 최적화 이론
│   ├── Discrete/              # 그래프 이론
│   └── Reference/             # 용어집 및 참고자료
├── Programming/               # 프로그래밍 기초
│   ├── 파이썬 기초.md
│   └── 자료구조와 알고리즘.md
├── ML Foundations/            # 머신러닝 기초 개념
│   ├── 학습 패러다임.md
│   ├── 손실 함수와 평가 지표.md
│   ├── 머신러닝 최적화.md
│   └── 정규화와 일반화.md
├── Operators/                 # 핵심 연산자
│   ├── 선형 연산자.md
│   ├── 행렬 분해.md
│   └── 합성곱 연산자.md
├── AI 기초 개요.md            # 전체 학습 로드맵
└── CLAUDE.md                  # 노트 작성 가이드라인
```

## 🎯 학습 로드맵

### 1단계: 수학 기초
**핵심 질문**: 선형대수·미적분·확률이 왜 필요한가?
- 시작: `Math/Linear Algebra/선형대수 핵심 개요.md`
- 필수 개념: 벡터, 행렬, 고유값, 미분, 그래디언트
- 활용: 신경망의 연산, 역전파, 파라미터 업데이트

### 2단계: 확률/통계·최적화
**핵심 질문**: 불확실성과 손실 최소화를 어떻게 다루나?
- 시작: `Math/Probability/확률과 통계 입문.md`
- 필수 개념: 확률분포, 기댓값, 경사하강법, 볼록 최적화
- 활용: 확률적 모델링, 손실 함수 최소화

### 3단계: 프로그래밍 기초
**핵심 질문**: Python과 자료구조를 어떻게 활용하나?
- 시작: `Programming/파이썬 기초.md`
- 필수 개념: NumPy, 자료구조, 알고리즘 복잡도
- 활용: 데이터 처리, 효율적 구현

### 4단계: ML 개념
**핵심 질문**: 지도/비지도 학습과 평가 지표는 무엇인가?
- 시작: `ML Foundations/학습 패러다임.md`
- 필수 개념: 훈련/검증/테스트, 손실 함수, 정확도/F1 스코어
- 활용: 모델 학습 및 평가

### 5단계: 최적화/정규화
**핵심 질문**: 학습 절차를 어떻게 안정화하나?
- 시작: `ML Foundations/머신러닝 최적화.md`
- 필수 개념: SGD, Adam, 배치 정규화, 드롭아웃
- 활용: 안정적이고 일반화된 모델 학습

### 6단계: 연산자/그래프
**핵심 질문**: 핵심 연산자와 계산 그래프는 어떻게 동작하나?
- 시작: `Operators/선형 연산자.md`
- 필수 개념: 선형 변환, 합성곱, 계산 그래프
- 활용: PyTorch 연산 이해, 자동 미분

## 📖 노트 구조

각 노트는 다음 템플릿을 따릅니다:

1. **TL;DR**: 3-5줄 핵심 요약
2. **핵심 개념**: 정의, 정리, 직관적 설명, 시각화
3. **수학적 전개**: 정리 증명 및 유도 (해당 시)
4. **예제**: 손으로 풀 수 있는 구체적 예제
5. **실습 예제**: Python/NumPy/PyTorch 코드
6. **연습 문제**: 난이도별 3-5개 문제
7. **참고 자료**: 교과서, 논문, 강의 링크
8. **다음 학습 경로**: 후속 노트 링크

자세한 작성 규칙은 `CLAUDE.md`를 참조하세요.

## 🚀 시작하기

### 방법 1: Obsidian으로 보기 (권장)
1. [Obsidian](https://obsidian.md/) 설치
2. 이 저장소를 클론
   ```bash
   git clone https://github.com/yourusername/ai_basics.git
   ```
3. Obsidian에서 "Open folder as vault" → `ai_basics` 폴더 선택
4. `AI 기초 개요.md`부터 시작

### 방법 2: GitHub에서 직접 보기
- 각 `.md` 파일은 GitHub에서도 렌더링됩니다
- `AI 기초 개요.md`에서 학습 로드맵 확인 후 필요한 노트로 이동

### 방법 3: 로컬에서 Markdown으로 보기
- VS Code + Markdown Preview Enhanced 플러그인
- 또는 Typora 등 Markdown 에디터 사용

## 📝 학습 전략

### Layered Approach
- 모르는 개념이 나오면 해당 단계 노트로 돌아가서 요약 확인
- 필요 시 참고 자료 섹션의 외부 링크 활용

### 실습 병행
- 수학 노트에서도 간단한 NumPy/PyTorch 코드로 개념 구현
- ML 노트에서는 실제 데이터셋으로 미니 프로젝트 수행

### 용어 정리
- `Math/Reference/개념 용어집.md`에서 용어 빠르게 조회
- 각 노트 말미의 참고 자료에서 심화 학습

## 🤝 기여 가이드

노트 개선 및 오류 수정을 환영합니다!

### 기여 방법
1. 이슈 등록: 오타, 수식 오류, 개선 제안
2. Pull Request: 새 노트 작성 또는 기존 노트 개선
   - `CLAUDE.md`의 작성 가이드라인 준수
   - 템플릿 구조 유지
   - 코드 예제는 실행 가능하도록 작성

### 작성 원칙
- ✅ 한국어 작성 (영문 용어는 병기)
- ✅ 수식 뒤 기호 설명 필수
- ✅ 실행 가능한 코드 제공
- ✅ 관련 노트와 양방향 링크 연결
- ✅ 맞춤법 및 수식 검증

자세한 내용은 `CLAUDE.md` 참조

## 📚 주요 노트 미리보기

### Math
- [선형대수 핵심 개요](Math/Linear%20Algebra/선형대수%20핵심%20개요.md)
- [미분과 미분법](Math/Calculus/미분과%20미분법.md)
- [경사하강법과 변형](Math/Optimization/경사하강법과%20변형.md)
- [확률과 통계 입문](Math/Probability/확률과%20통계%20입문.md)

### ML Foundations
- [학습 패러다임](ML%20Foundations/학습%20패러다임.md)
- [손실 함수와 평가 지표](ML%20Foundations/손실%20함수와%20평가%20지표.md)
- [머신러닝 최적화](ML%20Foundations/머신러닝%20최적화.md)

### Operators
- [선형 연산자](Operators/선형%20연산자.md)
- [합성곱 연산자](Operators/합성곱%20연산자.md)

## 📖 참고 자료

### 교과서
- **선형대수**: *Linear Algebra and Its Applications* (Gilbert Strang)
- **미적분**: *Calculus* (James Stewart)
- **확률**: *Introduction to Probability* (Bertsekas & Tsitsiklis)
- **딥러닝**: *Deep Learning* (Goodfellow, Bengio, Courville)

### 온라인 강의
- [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- [3Blue1Brown - Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Stanford CS229 - Machine Learning](https://see.stanford.edu/Course/CS229)

## 📄 라이선스

이 프로젝트는 교육 목적으로 자유롭게 사용 가능합니다.

## 📮 문의

- 이슈 등록: [GitHub Issues](https://github.com/yourusername/ai_basics/issues)
- 개선 제안: Pull Request 환영

---

**Happy Learning! 🎓**

*Last updated: 2025-11-03*
