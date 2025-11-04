## TL;DR
- AMP(Automatic Mixed Precision)는 연산을 FP16/FP32로 혼합해 속도를 높이고 메모리 사용을 줄입니다.
- Quantization은 정수 연산으로 모델을 경량화하며, Post-Training과 QAT 두 가지 접근이 있습니다.
- PyTorch 1.10+에서 `torch.cuda.amp`와 `torch.ao.quantization` 모듈을 활용하면 비교적 간단히 적용할 수 있습니다.

## 언제 쓰나
- GPU 메모리 여유가 없어 배치 크기를 늘리기 어려울 때 (AMP)
- 엣지 디바이스나 모바일에서 모델을 배포하려 할 때 (Quantization)
- 추론 속도를 향상시키면서 정확도 손실을 최소화하고 싶을 때

## AMP 핵심 흐름
|단계|코드|설명|
|---|---|---|
|autocast|`with autocast(): output = model(x)`|FP16/FP32 자동 선택|
|GradScaler|`scaler.scale(loss).backward()`|언더플로 방지|
|스케일 업데이트|`scaler.step(optimizer); scaler.update()`|optimizer 호출 전 필수|

### AMP 예제
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in loader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs.cuda())
        loss = criterion(outputs, targets.cuda())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Quantization 옵션
|방식|설명|장점|주의점|
|---|---|---|---|
|Post-Training Static|사전 학습된 모델에 Calibration 데이터로 정량화|구현 간단, 빠른 적용|정확도 손실 가능|
|Post-Training Dynamic|주로 활성화 정량화|RNN, NLP에 적합|속도 향상은 제한적|
|Quantization Aware Training (QAT)|학습 중 정량화 효과를 모사|정확도 유지|학습 비용 증가|

### QAT 간단 예
```python
import torch.ao.quantization as quant

model = MyModel()
model.fuse_model()
model.qconfig = quant.get_default_qat_qconfig("fbgemm")
quant.prepare_qat(model, inplace=True)

for epoch in range(5):
    train(model, loader)

quant.convert(model, inplace=True)
```

## 실수 주의
- AMP 사용 시 일부 연산(예: softmax, division)은 FP16에서 불안정할 수 있으므로 필요하면 `float()`로 캐스팅하세요.
- Quantization은 지원되는 연산만 가능하므로 모델 구조를 미리 확인하세요. 커스텀 연산은 대체 레이어가 필요합니다.
- Calibration 데이터는 실제 배포 데이터 분포와 유사해야 정확도 손실을 줄일 수 있습니다.
- QAT를 사용할 때는 학습률과 스케줄을 재조정해야 하며, 특정 파라미터가 학습되지 않는지 확인하세요.

## 관련 노트
- [[Advanced/Distributed and Parallel]]
- [[Training/Optimizers and Schedulers]]
- [[Deployment/Mobile and Edge]]
