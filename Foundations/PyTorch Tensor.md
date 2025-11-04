## TL;DR
- PyTorch Tensor는 수치 계산을 위해 GPU 가속이 가능한 핵심 데이터 구조입니다.
- 텐서 생성, 변환, 자동미분 연동 방법을 정리해 모델 구현 기반을 다집니다.

## 언제 쓰나
- 심층 신경망 모델에서 입력·파라미터·중간 결과를 표현할 때
- NumPy 배열 대비 GPU 연산과 autograd 지원이 필요할 때
- 데이터 전처리부터 추론 후처리까지 동일 API 흐름을 유지하고 싶을 때

## 주요 API
| 구분 | 함수 | 용도 | 메모 |
| --- | --- | --- | --- |
| 생성 | `torch.tensor`, `torch.zeros`, `torch.randn` | 데이터/노이즈 기반 텐서 초기화 | `dtype`, `device` 인자 확인 |
| 변환 | `tensor.view`, `tensor.reshape`, `tensor.permute` | 텐서 형태 재구성 | 연속성 여부에 따라 `view`/`reshape` 선택 |
| 장치 이동 | `tensor.to`, `tensor.cuda` | CPU↔GPU 전송 | 모델과 동일 장치 유지 필요 |
| 자동미분 | `tensor.requires_grad_` | 기울기 추적 활성화 | 필요 없는 텐서는 `with torch.no_grad()` |

## 실습 예제
```python
import torch

def describe_tensor(x: torch.Tensor, name: str) -> None:
    print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, requires_grad={x.requires_grad}")

if __name__ == "__main__":
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    noise = torch.randn_like(data) * 0.1
    combined = data + noise

    describe_tensor(data, "data")
    describe_tensor(noise, "noise")
    describe_tensor(combined, "combined")

    loss = combined.pow(2).mean()
    loss.backward()
    print("grad:\n", data.grad)
```

## 실수 주의
- NumPy 배열을 텐서로 변환할 때 복사 여부를 확인하고, 필요 시 `np.copy` 후 변환하세요.
- `view` 사용 전 텐서가 메모리 연속인지 `tensor.is_contiguous()`로 확인하거나 `tensor.contiguous()` 호출을 고려하세요.
- CPU와 GPU 텐서를 혼합 연산하면 오류가 발생하므로 연산 전 장치를 일치시킵니다.
- Autograd 추적이 필요 없는 지점에서는 `with torch.no_grad()`나 `tensor.detach()`로 그래프를 끊어 메모리를 절약하세요.

## 관련 노트
- [[Foundations/5. 신경망 기본 구조]]
- [[Programming/1. 파이썬 기초]]
- [[Foundations/3. 최적화]]
