## TL;DR
- 텐서 연산을 산술, 선형대수, 집계, 논리/조건, 기타 유틸 다섯 범주로 정리했습니다.
- 형태 변형, dtype 전환 등은 각각 [[Tensor 조작]], [[Tensor 변환]] 노트로 이동했습니다.
- 각 범주마다 대표 함수와 간단한 실습 코드를 담아 바로 테스트할 수 있습니다.

## 카테고리 개요
|카테고리|대표 함수|한 줄 요약|연관 노트|
|---|---|---|---|
|산술(Arithmetic)|`add`, `sub`, `mul`, `div`, `pow`|원소별 기본 연산|[[Tensor 생성]]|
|선형대수|`matmul`, `mm`, `bmm`, `einsum`, `linalg`|행렬·배치 연산|[[Tensor 가속]]|
|집계/통계|`sum`, `mean`, `std`, `max`, `argmax`|차원 축소와 통계량|[[Tensor 인덱싱]]|
|논리/조건|`eq`, `gt`, `logical_and`, `where`|조건 분기와 비교|[[Tensor 인덱싱]]|
|기타 유틸|`clamp`, `round`, `exp`, `log`, `abs`|값 제한과 스칼라 함수|[[Tensor 변환]]|

## 산술 연산
|함수|설명|기억 포인트|
|---|---|---|
|`torch.add(a, b)`|원소별 덧셈|in-place는 `a.add_(b)`|
|`torch.sub(a, b)`|원소별 뺄셈|브로드캐스팅 지원|
|`torch.mul(a, b)`|원소별 곱|스칼라 곱 가능|
|`torch.div(a, b)`|원소별 나눗셈|`rounding_mode` 선택 가능|
|`torch.pow(a, n)`|거듭제곱|텐서·스칼라 지수 모두 지원|
|`torch.remainder(a, b)`|나머지|부호 주의|

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([0.5, 1.0, 1.5])
print(torch.add(a, b))
print(torch.pow(a, 2))
```

## 선형대수 연산
|함수|설명|기억 포인트|
|---|---|---|
|`torch.matmul(a, b)`|일반 행렬곱|배치 차원 자동 브로드캐스트|
|`torch.mm(a, b)`|2D 전용 행렬곱|입력 2D 제한|
|`torch.bmm(a, b)`|배치 단위 행렬곱|shape `[B, N, M]`|
|`torch.einsum(eq, *tensors)`|Einstein 표기식 연산|복잡한 연산 패턴 표현|
|`torch.linalg.inv(a)`|역행렬|비가역 행렬 주의|
|`torch.linalg.svd(a)`|SVD 분해|U, S, V 반환|

```python
import torch

x = torch.randn(32, 128)
w = torch.randn(128, 64)
y = torch.matmul(x, w)

queries = torch.randn(2, 16, 64)
keys = torch.randn(2, 32, 64)
attn = torch.einsum('bqd,bkd->bqk', queries, keys)
```

## 집계·통계 연산
|함수|설명|기억 포인트|
|---|---|---|
|`torch.sum(x, dim=None, keepdim=False)`|합|`keepdim`으로 차원 유지|
|`torch.mean(x, dim)`|평균|dtype 자동 승격|
|`torch.std(x, dim, unbiased=True)`|표준편차|훈련 시 `False` 자주 사용|
|`torch.max(x, dim)`|최댓값과 인덱스|`values, indices` 반환|
|`torch.argmax(x, dim)`|최댓값 인덱스|분류 예측|
|`torch.cumsum(x, dim)`|누적 합|시계열 누적 계산|

```python
import torch

logits = torch.randn(4, 10)
probs = logits.softmax(dim=1)
entropy = -(probs * probs.log()).sum(dim=1)

running = torch.cumsum(torch.ones(5), dim=0)
print(entropy, running)
```

## 논리·조건 연산
|함수|설명|기억 포인트|
|---|---|---|
|`torch.eq(a, b)`·`ne`·`gt` 등|원소별 비교|bool 텐서 반환|
|`torch.logical_and(a, b)`|논리 AND|bool dtype 필요|
|`torch.where(cond, a, b)`|조건에 따라 선택|브로드캐스팅 가능|
|`torch.isfinite(x)`|유한 값 검사|NaN/Inf 필터링|
|`torch.all(x, dim)`·`torch.any(x, dim)`|조건 요약|bool 축소|

```python
import torch

scores = torch.tensor([0.2, 0.9, float('inf')])
mask = torch.isfinite(scores) & (scores > 0.5)
filtered = torch.where(mask, scores, torch.zeros_like(scores))
print(mask, filtered)
```

## 기타 유틸 / 스칼라 함수
|함수|설명|기억 포인트|
|---|---|---|
|`torch.clamp(x, min, max)`|값 범위 제한|분석 모델 안정화|
|`torch.round`, `floor`, `ceil`|반올림·내림·올림|정수 변환 전 사용|
|`torch.exp`, `log`|지수/로그|Softmax 구성 요소|
|`torch.abs(x)`|절댓값|손실 균형 조정|
|`torch.sigmoid`, `tanh`, `relu`|비선형 활성화|`torch.nn.functional`과 동일|

```python
import torch

raw = torch.linspace(-2, 2, steps=5)
bounded = torch.clamp(raw, min=-0.5, max=0.5)
activations = torch.sigmoid(raw)
print(bounded, activations)
```

## 중복 이동 안내
- 형태 변형 관련 함수(`reshape`, `permute`, `cat`, `stack`)는 [[Tensor 조작]]에서 자세히 다룹니다.
- dtype·device 전환(`to`, `cuda`, `cpu`, `detach`)은 [[Tensor 변환]]으로 이동했습니다.

## 관련 노트
- [[Tensor 조작]]
- [[Tensor 변환]]
- [[Tensor 인덱싱]]
