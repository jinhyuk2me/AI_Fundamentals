## TL;DR
- PyTorch 전반을 빠르게 훑고, 각 세부 노트가 어떤 흐름으로 이어지는지 보여주는 허브입니다.
- 설치 확인 → Tensor 기초 → 모델 구성 → 학습 루프 → 평가/배포 순으로 학습 경로를 제안합니다.
- 실습과 프로젝트 노트로 연결해 이론 정리와 실전 경험을 동시에 쌓을 수 있도록 설계했습니다.

## 학습 로드맵
| 단계 | 주제 | 핵심 질문 | 바로 가기 | 다음 이동 |
| --- | --- | --- | --- | --- |
| 1 | Tensor 기초 | PyTorch 텐서는 어떻게 다룰까? | [[Tensor Hub]] | [[Modeling/Module Basics]] |
| 2 | 데이터 준비 | Dataset·DataLoader를 어떻게 구성할까? | [[Data/Dataset and DataLoader]] | [[Data/IO and Storage]] |
| 3 | 모델 구성 | 레이어와 모듈을 어떻게 조합할까? | [[Modeling/Module Basics]] | [[Modeling/Core Layers]] |
| 4 | 학습 루프 | 학습/검증 루프와 옵티마이저는 어떻게 설계할까? | [[Training/Training Loop Patterns]] | [[Training/Optimizers and Schedulers]] |
| 5 | 평가·모니터링 | 모델 성능을 어떻게 측정하고 기록할까? | [[Evaluation/Metrics]] | [[Evaluation/Logging and Visualization]] |
| 6 | 고급 최적화 | 분산 학습·AMP·디버깅은 어떻게 적용할까? | [[Advanced/Distributed and Parallel]] | [[Advanced/Profiling and Debugging]] |
| 7 | 배포 | 학습한 모델을 어디서 어떻게 서빙할까? | [[Deployment/Serving Options]] | [[Deployment/Mobile and Edge]] |
| 8 | 생태계 활용 | PyTorch 주변 도구는 어떻게 활용할까? | [[Ecosystem/Lightning and Utilities]] | [[Ecosystem/Project Tooling]] |
| 9 | 실전 프로젝트 | 배운 내용으로 파이프라인을 어떻게 완성할까? | [[Projects/Project Template]] | [[Projects/Image Classification Pipeline]] |

## 설치 및 환경 점검
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
x = torch.randn(2, 3, device='cuda' if torch.cuda.is_available() else 'cpu')
print(x.dtype, x.device)
```
- CUDA 버전과 PyTorch 빌드가 맞는지 확인하고, GPU가 없다면 CPU 빌드를 설치합니다.
- reproducibility가 필요하면 `torch.manual_seed(42)`와 `torch.backends.cudnn.deterministic = True` 등을 설정합니다.

## PyTorch 핵심 모듈 한눈에
| 모듈 | 역할 | 노트 |
| --- | --- | --- |
| `torch` | Tensor 연산, 수학 함수, 랜덤 시드 | [[Tensor Hub]] |
| `torch.nn` | 모델 구성 모듈, 파라미터 관리 | [[Modeling/torch.nn/torch.nn Overview]] |
| `torch.optim` | 옵티마이저, 학습 스케줄러 | [[Training/Optimizers and Schedulers]] |
| `torch.utils.data` | 데이터셋, 로딩 파이프라인 | [[Data/Dataset and DataLoader]] |
| `torch.autograd` | 자동 미분 엔진 | [[Tensor Autograd]] |
| `torch.cuda` | GPU 디바이스 관리 | [[Tensor 가속]] |

## 학습 전략
- **이론 → 실습 → 프로젝트**: 각 노트를 읽고 나면 작은 코드 스니펫으로 실습 후, `Projects` 디렉터리의 미니 파이프라인으로 확장합니다.
- **링크 활용**: 노트 말미 “관련 노트” 링크를 따라가며 부족한 개념을 보완하면 탐색 시간이 줄어듭니다.
- **버전 기록**: 노트 업데이트 시 날짜와 변경 요약을 남기면 재학습 시 추적이 쉽습니다.

## 추천 과제
- `[[Projects/Image Classification Pipeline]]`: Tensor·모델·학습 루프·평가까지 한 번에 다루는 기본 프로젝트.
- `[[Projects/Time Series Forecasting]]`: 시계열 특화 전처리와 RNN/Transformer 모델 구축 실습.
- `[[Projects/NLP Pipeline]]`: 토큰화, 임베딩, 시퀀스 모델링, 평가까지 자연어 처리 파이프라인 구성.

## 관련 노트
- [[PyTorch vs Others]]
- [[Tensor Hub]]
- [[Modeling/Module Basics]]
