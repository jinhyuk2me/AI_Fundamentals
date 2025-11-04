## TL;DR
- 성능 병목과 학습 이상을 파악하려면 프로파일러, gradient 검사, 디버깅 툴을 적극 활용해야 합니다.
- `torch.profiler`, `autograd.detect_anomaly`, `register_hook`, logging 도구로 문제를 재현·진단합니다.
- 디버깅 시 reproducibility를 위해 시드를 고정하고, 실험 환경을 기록합니다.

## 언제 쓰나
- 학습 속도가 비정상적으로 느리거나 메모리 사용량이 급증할 때
- Loss가 갑자기 NaN/Inf가 되거나 gradient 폭주가 의심될 때
- 모델 구조 변경 후 예상과 다른 출력이 나올 때

## 주요 도구
|도구|설명|용도|
|---|---|---|
|`torch.profiler.profile`|CPU/GPU 시간, 연산 트레이스 분석|성능 병목 파악|
|`torch.autograd.detect_anomaly`|NaN/Inf gradient 위치 추적|수치 오류 디버깅|
|`register_forward_hook`/`register_backward_hook`|중간 텐서 확인|shape/gradient 검사|
|`torch.cuda.memory_summary()`|GPU 메모리 사용 현황|메모리 누수 확인|
|`torch.set_printoptions`, `torch.set_grad_enabled`|출력/gradient 제어|디버깅 편의성|

## 프로파일링 예제
```python
import torch
from torch import profiler

model = MyModel().cuda()
inputs = torch.randn(32, 3, 224, 224, device="cuda")

with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA],
        record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        for _ in range(10):
            model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Gradient 디버깅
```python
torch.autograd.set_detect_anomaly(True)

for batch in loader:
    optimizer.zero_grad()
    outputs = model(batch["x"].to(device))
    loss = criterion(outputs, batch["y"].to(device))
    loss.backward()
    optimizer.step()
```

## 실수 주의
- 프로파일러를 장시간 켜 두면 오버헤드가 커집니다. 필요한 구간만 프로파일링하고 종료하세요.
- `detect_anomaly`는 속도를 크게 낮추므로 디버깅 시에만 사용하고 학습에는 비활성화하세요.
- hook 사용 시 메모리 참조가 유지되어 GC가 지연될 수 있으니, 사용 후 반드시 제거(`handle.remove()`)하세요.
- NaN/Inf가 발생하면 입력 데이터, 손실 함수, optimizer 설정 등을 모두 점검해야 합니다.

## 관련 노트
- [[Advanced/AMP and Quantization]]
- [[Training/Training Loop Patterns]]
- [[Evaluation/Logging and Visualization]]
