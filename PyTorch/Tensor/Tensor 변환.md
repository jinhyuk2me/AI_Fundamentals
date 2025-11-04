## TL;DR
- dtype, device, 메모리 공유 여부를 제어하는 함수들을 한곳에 정리했습니다.
- `.to()`는 dtype과 device를 동시에 바꿀 수 있는 만능 도구입니다.
- Autograd 그래프에서 분리하거나 완전한 복사본을 만들 때는 `detach`와 `clone`의 조합을 기억하세요.

## 언제 쓰나
- 데이터 전처리 후 모델이 기대하는 dtype·device로 맞출 때
- 추론 시 gradient 추적을 끄거나 파라미터 스냅샷을 만들 때
- NumPy와 PyTorch, CPU와 GPU 사이를 오갈 때

## 주요 API
|함수|설명|기억 포인트|
|---|---|---|
|`tensor.to(dtype=..., device=...)`|dtype·device 동시 변환|매개변수 생략 가능|
|`tensor.type(dtype)`|dtype만 변환|가급적 `.to(dtype=...)` 사용|
|`tensor.float()` / `.long()` 등|자주 쓰는 dtype 단축|Autograd 그래프 유지|
|`tensor.cuda()` / `.cpu()`|장치 이동 단축|`.to('cuda')`가 더 일반적|
|`tensor.detach()`|그래프에서 분리|데이터 공유, gradient X|
|`tensor.clone()`|깊은 복사|그래프 유지|
|`tensor.detach().clone()`|독립 복사|학습 스냅샷 생성|
|`torch.from_numpy(nd)` / `tensor.numpy()`|NumPy ↔ Tensor|CPU 텐서만 변환 가능|

## 실습 예제
```python
import torch
import numpy as np

x = torch.randn(4, 4, requires_grad=True)

x_fp16 = x.to(dtype=torch.float16, device='cuda')
x_cpu = x_fp16.to('cpu', dtype=torch.float32)

with torch.no_grad():
    pred = x_cpu.softmax(dim=-1)

arr = np.array([1, 2, 3], dtype=np.float32)
t = torch.from_numpy(arr)                  # 메모리 공유
t_clone = t.clone().detach()               # 완전 독립 복사
```

## 실수 주의
- `tensor.numpy()`는 CPU 텐서에서만 동작합니다. GPU → CPU 이동 후 변환하세요.
- `detach()`만 호출하면 원본과 메모리를 공유하므로 수정 시 함께 변합니다. 필요하면 `.clone()`까지 사용하세요.
- dtype 변환 후 정밀도가 달라질 수 있습니다. 특히 `float16`은 혼합 정밀도(AMP)와 함께 사용하는 것이 안전합니다.

## 관련 노트
- [[Tensor 생성]]
- [[Tensor 가속]]
- [[Tensor Autograd]]
