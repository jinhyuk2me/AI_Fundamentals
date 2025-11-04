## TL;DR
- 텐서는 기존 데이터, 고정 값, 난수, 시퀀스 등 다양한 초기화 방법으로 만들 수 있습니다.
- `_like` 계열과 NumPy 연동을 익히면 기존 텐서를 손쉽게 재활용할 수 있습니다.
- 생성 직후 dtype과 device를 반드시 확인하고 필요한 경우 `.to()`로 정리합니다.

## 언제 쓰나
- 모델 입력용 더미 데이터를 만들어 실험하거나 디버깅할 때
- 초기 파라미터, 마스크, 인덱스 텐서를 준비할 때
- NumPy 또는 파이썬 리스트 데이터를 PyTorch 파이프라인으로 옮길 때

## 주요 API

### 기본 생성
|함수|설명|기억 포인트|
|---|---|---|
|`torch.tensor(data, dtype=None, device=None)`|파이썬 객체·NumPy에서 텐서 생성|데이터를 복사하므로 dtype 명시 추천|
|`torch.zeros(size)` / `torch.ones(size)` / `torch.full(size, value)`|고정 값으로 채우기|GPU는 `device=`로 지정|
|`torch.eye(n, m=None)`|단위 행렬 생성|정방/직사각 모두 가능|

### 난수 생성
|함수|설명|기억 포인트|
|---|---|---|
|`torch.rand(size)`|[0, 1) 균등분포|재현성 위해 `manual_seed` 사용|
|`torch.randn(size)`|평균 0, 표준편차 1 정규분포|가중치 초기화 기본값|
|`torch.randint(low, high, size)`|정수 균등분포|상한 `high` 미포함|
|`torch.normal(mean, std, size)`|사용자 정의 정규분포|`mean`·`std`는 스칼라/텐서 모두 허용|

### 시퀀스·패턴 생성
|함수|설명|기억 포인트|
|---|---|---|
|`torch.arange(start, end, step)`|등차수열|실수 step 사용 시 부동소수 오차 주의|
|`torch.linspace(start, end, steps)`|구간 균등 분할|양 끝 포함|
|`torch.logspace(start, end, steps)`|로그 스케일 분할|지수적 증가 값 생성|

### 기존 텐서 재사용
| 함수                                           | 설명             | 기억 포인트                |
| -------------------------------------------- | -------------- | --------------------- |
| `torch.empty(size)`                          | 메모리만 확보(값 미정)  | 바로 `uniform_` 등으로 채우기 |
| `torch.zeros_like(t)` / `torch.rand_like(t)` | shape 복사       | dtype/device 자동 상속    |
| `torch.from_numpy(nd)` / `tensor.numpy()`    | NumPy ↔ Tensor | CPU 텐서만 NumPy 변환 가능   |
| `tensor.clone()`                             | 깊은 복사          | Autograd 그래프 유지       |

## 실습 예제
```python
import torch
import numpy as np

torch.manual_seed(42)

grid = torch.arange(0.0, 1.0, 0.2).view(1, -1)
noise = torch.randn(3, 5)

x = torch.linspace(-1, 1, steps=5, dtype=torch.float64)
y = torch.full((3, 5), 7, device='cpu')

arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
t = torch.from_numpy(arr)          # 메모리 공유
t_clone = t.clone()                # 독립 복사

dummy_input = torch.randn(32, 3, 224, 224, device='cuda', dtype=torch.float32)
```

## 실수 주의
- `torch.tensor(np_array)`는 NumPy dtype을 그대로 따라가므로 필요 시 `.float()` 등으로 명시하세요.
- `torch.empty`는 초기화되지 않은 값을 담고 있으니 사용 전에 값을 채워야 합니다.
- GPU 텐서는 생성 시 `device='cuda'`로 바로 만들면 `.to()` 호출보다 빠릅니다.

## 관련 노트
- [[Tensor 조작]]
- [[Tensor 변환]]
- [[Tensor 입출력]]
