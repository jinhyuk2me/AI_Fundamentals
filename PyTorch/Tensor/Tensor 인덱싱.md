## TL;DR
- 인덱싱은 텐서의 일부 값만 선택하거나 재배열할 때 사용합니다.
- 기본 인덱싱, 슬라이싱, 불리언 마스크, 고급 인덱싱, `gather` 계열을 구분해 사용하면 대부분의 추출 작업을 다룰 수 있습니다.
- 고급 인덱싱은 항상 새 텐서를 반환하므로 메모리와 Autograd 영향에 주의하세요.

## 언제 쓰나
- 배치에서 특정 샘플이나 특징만 추출할 때
- 조건에 맞는 요소만 골라 손실을 계산할 때
- 시퀀스를 정렬하거나 포인터 기반 데이터를 모을 때

## 주요 패턴
|패턴|예시|설명|
|---|---|---|
|기본 인덱싱|`x[0, 2]`|단일 위치 접근, view|
|슬라이싱|`x[:, 1:4]`|연속 구간 선택, view|
|불리언 마스크|`x[x > 0]`|조건에 맞는 원소만 선택, 복사|
|고급 인덱싱|`x[[0, 2], [1, 3]]`|지정 좌표 추출, 복사|
|`torch.gather`|`torch.gather(x, dim, idx)`|인덱스 텐서를 따라 값 수집|
|`torch.take_along_dim`|`torch.take_along_dim(x, idx, dim)`|차원별 값 추출, gather 대안|

## 실습 예제
```python
import torch

scores = torch.tensor([[0.1, 0.8, 0.1],
                       [0.4, 0.3, 0.3]])

top1 = scores.argmax(dim=1)               # 텐서([1, 0])
positive = scores[scores > 0.3]           # 조건 필터링

x = torch.arange(24).reshape(2, 3, 4)
channels_1_2 = x[:, :, 1:3]               # 슬라이스(view)

idx = torch.tensor([[0, 2], [1, 3]])
gathered = torch.gather(x, 2, idx)        # 고급 인덱싱 대체

print(top1, positive, channels_1_2.shape, gathered.shape)
```

## 실수 주의
- 불리언 마스크는 항상 새 텐서를 반환하므로 대규모 데이터에서는 `torch.where`나 `masked_fill_` 같은 in-place 방법도 고려하세요.
- 슬라이싱은 view이므로 원본 수정에 영향을 줍니다. 복사본이 필요하면 `.clone()`을 호출하세요.
- 반복적인 고급 인덱싱은 성능을 떨어뜨릴 수 있습니다. 가능한 경우 `gather`나 `scatter_`로 변환하세요.

## 관련 노트
- [[Tensor 조작]]
- [[Tensor 연산]]
- [[Tensor Autograd]]
