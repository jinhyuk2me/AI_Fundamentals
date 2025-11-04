## TL;DR
- PyTorch 2.0 이후 `torch.compile`을 비롯한 그래프 기반 최적화가 도입되어 실행 속도와 효율이 개선되었습니다.
- TorchScript, ONNX 변환은 모델을 배포 환경에 적합한 형태로 저장·최적화할 때 사용합니다.
- 각 방식마다 지원 연산과 제약이 다르므로 목표 환경에 맞춰 선택해야 합니다.

## 언제 쓰나
- 모델 추론 속도를 향상시키거나 벡엔드별 최적화를 적용하고 싶을 때
- Python 인터프리터 의존성을 줄이고 모델을 독립적으로 실행해야 할 때
- 다른 프레임워크나 런타임(ONNX Runtime, TensorRT)으로 모델을 내보낼 때

## 옵션 비교
|기술|설명|장점|주의점|
|---|---|---|---|
|`torch.compile`|동적 그래프를 컴파일해 빠르게 실행|코드 변경 최소|지원되지 않는 연산이 있을 수 있음|
|TorchScript|`trace` 또는 `script`로 그래프 생성|C++ 배포 가능|동적 제어 흐름 제약|
|ONNX Export|ONNX 포맷으로 변환|다양한 런타임 호환|export 실패 시 대체 연산 필요|

## torch.compile 예제
```python
import torch

model = MyModel().cuda()
compiled_model = torch.compile(model, mode="max-autotune")

inputs = torch.randn(32, 3, 224, 224, device="cuda")
with torch.inference_mode():
    output = compiled_model(inputs)
```

## TorchScript 예제
```python
import torch

model = MyModel().eval()
scripted = torch.jit.script(model)
scripted.save("model.pt")

loaded = torch.jit.load("model.pt")
with torch.no_grad():
    loaded(torch.randn(1, 3, 224, 224))
```

## ONNX Export 예제
```python
import torch

model = MyModel().eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
```

## 실수 주의
- `torch.compile`은 PyTorch 2.x에서 안정화되고 있으나, 커스텀 연산이나 일부 동적 형태는 지원되지 않을 수 있습니다.
- TorchScript `trace`는 데이터 의존 제어 흐름을 캡처하지 못하므로, 복잡한 로직은 `script`를 사용하세요.
- ONNX 변환은 opset 버전과 지원 연산에 따라 실패할 수 있으니, 변환 전 도표(`torch.onnx.export` 로그)를 확인하고 필요한 경우 opset을 낮추거나 대체 구현을 작성하세요.
- 그래프 기반 최적화를 적용한 모델은 반드시 정확도와 성능을 재검증하세요. 일부 최적화는 수치적 차이를 야기할 수 있습니다.

## 관련 노트
- [[Deployment/Serving Options]]
- [[Deployment/Mobile and Edge]]
- [[Projects/Project Template]]
