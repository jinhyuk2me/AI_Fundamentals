## TL;DR
- 학습 루프는 데이터 로딩 → 순전파 → 손실 계산 → 역전파 → 최적화 순으로 반복됩니다.
- Epoch/Batch 제어, gradient 누적, mixed precision, gradient clipping 같은 패턴을 상황에 맞게 적용합니다.
- 검증 루프와 로그 수집을 동일 구조 안에서 관리하면 유지보수가 용이합니다.

## 언제 쓰나
- 새로운 모델을 학습시키기 위한 기본 루프를 작성할 때
- 여러 모델을 비교하거나 실험을 자동화할 공통 루프가 필요할 때
- 분산 학습, mixed precision 등을 통합하고자 할 때

## 대표 패턴
|패턴|설명|핵심 코드|관련 노트|
|---|---|---|---|
|기본 학습 루프|순전파 → 손실 → 역전파 → 최적화|`loss.backward()`, `optimizer.step()`|[[Modeling/Module Basics]]|
|Gradient 누적|메모리 절약을 위해 여러 배치의 gradient를 합산 후 업데이트|`loss / accum_steps`|[[Advanced/Profiling and Debugging]]|
|Mixed Precision|FP16/FP32 혼합으로 속도 향상|`torch.cuda.amp.autocast`, `GradScaler`|[[Advanced/AMP and Quantization]]|
|검증 루프|`torch.no_grad()`로 추론|`model.eval()`, `metric.update`|[[Evaluation/Metrics]]|
|로깅/체크포인트|주기적 로그, 모델 저장|`scheduler.step()`, `torch.save`|[[Evaluation/Logging and Visualization]]|

## 실습 예제
```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train(model, train_loader, val_loader, criterion, optimizer, device):
    scaler = GradScaler(enabled=device.type == "cuda")
    best_val_loss = float("inf")

    for epoch in range(1, 6):
        model.train()
        for batch in train_loader:
            inputs = batch["x"].to(device, non_blocking=True)
            targets = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        val_loss = evaluate(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict()},
                       "best.pt")

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["x"].to(device, non_blocking=True)
            targets = batch["y"].to(device, non_blocking=True)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
    return total_loss / len(loader)
```

## 실수 주의
- `optimizer.zero_grad()` 호출을 빼먹으면 gradient가 누적되어 학습이 불안정해집니다.
- `model.eval()`과 `torch.no_grad()`를 검증/추론에서 함께 사용하지 않으면 BatchNorm, Dropout이 잘못 동작할 수 있습니다.
- Mixed precision 사용 시 손실을 바로 `backward()`하면 언더플로가 발생할 수 있으므로 반드시 `GradScaler`를 사용하세요.
- 분산 학습에서는 `DistributedSampler`와 `sampler.set_epoch()`를 사용해 셔플을 일관되게 유지해야 합니다.

## 관련 노트
- [[Training/Optimizers and Schedulers]]
- [[Training/Loss Functions]]
- [[Evaluation/Metrics]]
