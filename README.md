# AI Fundamentals

AI/딥러닝 기초 학습을 위한 Obsidian Vault입니다. 수학, 프로그래밍, 머신러닝 개념을 단계별로 정리했습니다.

## 특징

- 수학 이론과 Python/PyTorch 실습 코드를 함께 제공
- 정의 → 예제 → 실습 → 연습문제 순서로 구성
- Obsidian 링크로 관련 개념 연결
- 한국어 작성 (영문 용어 병기)

## 사전 준비

이 저장소는 기본적인 Python 지식을 가정합니다. Python을 처음 시작한다면 다음 리소스를 추천합니다:
- **Python 기초**: [Real Python](https://realpython.com/) 또는 [Python 공식 튜토리얼](https://docs.python.org/3/tutorial/)
- **NumPy**: [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- **환경 설정**: Anaconda 또는 `venv`로 가상환경 구성

## 저장소 구조

```
ai_fundamentals/
├── 0. AI 분류 체계.md         # AI 전체 개념 지도
├── AI 기초 개요.md            # 학습 로드맵 (시작은 여기서!)
├── Math/                      # 수학 기초
│   ├── Linear Algebra/        # 선형대수
│   ├── Calculus/              # 미적분
│   ├── Probability/           # 확률과 통계
│   ├── Optimization/          # 최적화 이론
│   └── ...
├── MLDL Basic/                # ML/DL 기초 개념 및 구현
├── PyTorch/                   # PyTorch 실습 노트
└── CLAUDE.md                  # 노트 작성 가이드라인
```

## 학습 시작하기

### 1. AI 전체 지형도 파악
**[0. AI 분류 체계.md](0.%20AI%20분류%20체계.md)** 에서 AI의 전체 개념 지도를 확인하세요.
- Rule-Based AI, Machine Learning, Deep Learning의 관계
- 학습 패러다임 (지도/비지도/강화 등)
- 실무 문제 해결 접근법

### 2. 학습 로드맵 따라가기
**[AI 기초 개요.md](AI%20기초%20개요.md)** 에서 단계별 학습 경로를 확인하세요.
- 9단계로 구성된 학습 로드맵
- 각 단계별 핵심 질문과 필수 개념
- 학습 전략 및 실습 가이드

## 설치 및 사용

### Obsidian으로 보기 (권장)
1. [Obsidian](https://obsidian.md/) 설치
2. 저장소 클론
   ```bash
   git clone https://github.com/yourusername/ai_fundamentals.git
   ```
3. Obsidian에서 "Open folder as vault" → `ai_fundamentals` 폴더 선택
4. `AI 기초 개요.md`에서 학습 시작

### 다른 방법
- **GitHub**: 웹에서 각 파일 탐색
- **로컬 에디터**: VS Code, Typora 등 Markdown 에디터 사용

## 노트 구조

각 노트는 다음 구조로 작성됩니다:

1. **TL;DR**: 핵심 요약 (3-5줄)
2. **핵심 개념**: 정의, 정리, 직관적 설명
3. **수학적 전개**: 증명 및 유도 (필요시)
4. **예제**: 손으로 풀 수 있는 작은 예제
5. **실습 예제**: 실행 가능한 Python/PyTorch 코드
6. **연습 문제**: 난이도별 3-5개
7. **참고 자료**: 교과서, 논문, 강의 링크
8. **다음 학습 경로**: 관련 노트 링크

## 기여

오류 수정이나 개선 제안은 이슈나 PR로 남겨주세요.

### 작성 원칙
- 한국어 작성 (영문 용어 병기)
- 수식 뒤 기호 설명
- 실행 가능한 코드 제공
- 관련 노트와 링크 연결

자세한 작성 가이드라인은 [CLAUDE.md](CLAUDE.md)를 참고하세요.

## 라이선스

이 프로젝트는 교육 목적으로 자유롭게 사용 가능합니다.

## 문의

- 이슈 등록: [GitHub Issues](https://github.com/yourusername/ai_fundamentals/issues)
- 개선 제안: Pull Request 환영

---

*Last updated: 2025-11-04*
