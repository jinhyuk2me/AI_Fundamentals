## TL;DR
- 텐서 조작은 모양을 맞추고 차원을 재배열하는 작업으로 전처리와 모델 입출력 사이를 잇습니다.
- 차원 추가/제거, 결합/분할, 브로드캐스팅을 익히면 대부분의 데이터 흐름을 다룰 수 있습니다.
- `view`와 `reshape`의 차이, `contiguous` 처리 여부를 항상 확인하세요.

## 언제 쓰나
- CNN 입력을 `NHWC ↔ NCHW`로 바꿀 때
- 시계열 데이터를 배치 단위로 펼치거나 합칠 때
- 여러 텐서를 batch 축으로 연결하거나 일정 길이로 분할할 때

## 주요 API

### 형태 변형
|함수|설명|기억 포인트|
|---|---|---|
|`tensor.view(*shape)`|연속 메모리에서 모양 변경|`is_contiguous()` 확인|
|`tensor.reshape(*shape)`|필요 시 복사 후 모양 변경|안전하지만 약간 느릴 수 있음|
|`tensor.flatten(start_dim=1)`|다차원 → 1차원|Batch 차원 제외 flatten|

### 차원 조정
|함수|설명|기억 포인트|
|---|---|---|
|`tensor.unsqueeze(dim)`|새 차원 추가|배치/채널 맞출 때 유용|
|`tensor.squeeze(dim=None)`|크기 1 차원 제거|명시적 `dim` 사용으로 안전|
|`tensor.transpose(dim0, dim1)`|두 차원 교환|2D 이상에서 사용|
|`tensor.permute(*dims)`|차원 순서 재배열|입출력 포맷 변경|

### 결합·분할
|함수|설명|기억 포인트|
|---|---|---|
|`torch.cat(tensors, dim)`|차원 유지하며 이어 붙이기|연결 축 제외 shape 동일|
|`torch.stack(tensors, dim)`|새 차원 기준으로 쌓기|차원 수 +1|
|`torch.chunk(t, chunks, dim)`|균등 분할|마지막 chunk는 작을 수 있음|
|`torch.split(t, split_size, dim)`|지정 크기 분할|리스트 반환|

### 브로드캐스팅·반복
|함수|설명|기억 포인트|
|---|---|---|
|`tensor.expand(*sizes)`|복사 없이 view 확장|읽기 전용처럼 사용|
|`tensor.repeat(*sizes)`|실제 값 복제|메모리 증가 주의|
|`tensor.expand_as(other)`|다른 텐서 shape로 확장|Loss 계산 시 자주 사용|

### 정렬·회전
|함수|설명|기억 포인트|
|---|---|---|
|`torch.sort(t, dim, descending)`|값과 인덱스 반환|튜플 `(values, indices)`|
|`torch.topk(t, k, dim)`|상위 k개 선택|분류 결과 후처리|
|`torch.flip(t, dims)`|축 뒤집기|데이터 증강에 활용|
|`torch.roll(t, shifts, dims)`|순환 이동|시계열 윈도우 이동|

## 실습 예제
```python
import torch

x = torch.arange(24).reshape(2, 3, 4)
nhwc = x.permute(0, 2, 1)          # [N, H, W, C]
batch_flat = x.flatten(start_dim=1)

a = torch.ones(2, 3)
b = torch.zeros(2, 3)
ab = torch.cat([a, b], dim=1)

splits = torch.split(torch.arange(10), [4, 4, 2])
rolled = torch.roll(ab, shifts=1, dims=1)

print(nhwc.shape, batch_flat.shape, len(splits), rolled)
```

## 실수 주의
- `view` 사용 전 `tensor.is_contiguous()`를 확인하고 필요하면 `tensor.contiguous()`를 호출하세요.
- `stack`은 입력 텐서 shape가 완전히 같아야 합니다. 다르면 사전에 `reshape`로 맞추세요.
- `expand`는 실제 데이터를 복제하지 않으므로 수정이 필요한 경우 `clone()`으로 복사하세요.

## 관련 노트
- [[Tensor 생성]]
- [[Tensor 인덱싱]]
- [[Tensor 연산]]
