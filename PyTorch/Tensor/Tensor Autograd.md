## TL;DR
- Autograd는 연산 그래프를 기록해 `backward()` 호출 시 자동으로 gradient를 계산합니다.
- `requires_grad`, `no_grad`, `detach`로 추적 범위를 제어합니다.
- 여러 배치에서 gradient가 누적되므로 업데이트 전 `optimizer.zero_grad()`를 잊지 마세요.

## 언제 쓰나
- 신경망 학습에서 Loss를 기준으로 파라미터를 업데이트할 때
- 사전 학습된 특징을 고정하거나 일부 레이어만 미세 조정할 때
- Gradient를 직접 점검하거나 손수 계산할 때

## 주요 API
|함수/속성|설명|기억 포인트|
|---|---|---|
|`tensor.requires_grad_(True)`|Gradient 추적 활성화|in-place로 설정|
|`loss.backward()`|스칼라 loss 기준 역전파|loss는 스칼라여야 함|
|`tensor.grad`|Gradient 접근|backward 후 값 확인|
|`torch.autograd.grad(outputs, inputs)`|수동 gradient 계산|`create_graph` 옵션|
|`with torch.no_grad():`|블록 내 추적 비활성화|추론 시 필수|
|`tensor.detach()`|그래프에서 분리|데이터 공유|
|`torch.autograd.set_grad_enabled(flag)`|전역 토글|훈련/추론 전환|

## 실습 예제
```python
import torch

x = torch.randn(3, 3, requires_grad=True)
w = torch.nn.Linear(3, 1)

out = w(x).mean()
out.backward()

print(x.grad.shape)         # 입력 gradient
print(w.weight.grad)        # 파라미터 gradient

with torch.no_grad():
    inference = w(x)        # 추론 단계, gradient 추적 없음

features = x.detach()       # 그래프에서 분리된 텐서
```

## 실수 주의
- 텐서를 여러 번 backward하려면 `loss.backward(retain_graph=True)`가 필요합니다.
- Gradient는 기본적으로 누적됩니다. 루프마다 `optimizer.zero_grad()` 또는 `model.zero_grad()`를 호출하세요.
- Autograd가 추적 중일 때 `.item()`을 반복 호출하면 그래프가 끊기니 로그 출력에만 사용하세요.

## 관련 노트
- [[Tensor 변환]]
- [[Tensor 가속]]
- [[Tensor 연산]]
