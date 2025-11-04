## TL;DR
- Optimizer는 파라미터 업데이트 규칙을, Scheduler는 학습률 변화를 제어해 수렴 속도와 안정성을 높입니다.
- 모델·데이터 특성에 따라 SGD, Adam, AdamW 등을 선택하고 필요 시 weight decay, gradient clipping을 조합합니다.
- Scheduler는 warmup, step decay, cosine annealing 등을 활용해 학습 중 학습률을 조절합니다.

## 언제 쓰나
- 새로운 모델을 학습시킬 때 적절한 최적화 알고리즘을 선택해야 할 때
- 학습률 조정 전략을 통해 성능을 끌어올리고자 할 때
- fine-tuning, transfer learning 등에서 부분 파라미터만 업데이트하고 싶을 때

## Optimizer 비교
|옵티마이저|특징|장점|주의점|
|---|---|---|---|
|`SGD`|모멘텀, Nesterov 옵션 제공|일반화 성능 좋음, 메모리 사용 적음|학습률 조정 중요|
|`Adam`|적응형 학습률, 모멘텀 결합|빠른 수렴, 설정 쉬움|일반화가 다소 약할 수 있음|
|`AdamW`|Adam + decoupled weight decay|Transformer 계열 기본|weight decay 설정 주의|
|`RMSProp`|지수 이동 평균 기반 스케일링|RNN, 강화학습에서 사용|초기 학습률 민감|
|`Adagrad`|축적된 gradient로 학습률 조절|희소 피처에 유리|장기 학습 시 학습률 급감|

## Scheduler 비교
|스케줄러|설명|사용 시점|예시|
|---|---|---|---|
|`StepLR`|고정 에폭마다 학습률 감소|간단한 스텝 감소|`StepLR(optimizer, step_size=30, gamma=0.1)`|
|`MultiStepLR`|여러 에폭 지점에서 감소|fine-tuning|`milestones=[30, 60]`|
|`CosineAnnealingLR`|코사인 곡선으로 감소|Vision/Transformer|`T_max=100`|
|`ReduceLROnPlateau`|성능 정체 시 학습률 감소|검증 지표 기반|`mode='min', patience=3`|
|`OneCycleLR`|워밍업 후 감소|대규모 학습|`max_lr`, `pct_start`|

## 실습 예제
```python
import torch

model = torch.nn.Linear(128, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(1, 51):
    for batch in train_loader:
        inputs = batch["x"]
        targets = batch["y"]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()
    print(f"Epoch {epoch} lr={scheduler.get_last_lr()[0]:.6f}")
```

## 실수 주의
- `scheduler.step()` 호출 시점은 스케줄러 종류에 따라 다르므로 문서를 확인하세요. 일부는 배치 단위(`step()`), 일부는 에폭 단위(`epoch_end`)입니다.
- `ReduceLROnPlateau`는 `scheduler.step(metric)`처럼 모니터할 값이 필요합니다. 호출 순서를 잘못 설정하면 적용되지 않습니다.
- 파라미터 그룹을 나눌 때 `optimizer = Adam([{'params': backbone, 'lr': 1e-4}, {'params': head, 'lr': 1e-3}])`처럼 그룹마다 다른 옵션을 지정할 수 있지만 키 이름을 잊으면 기본값이 적용됩니다.
- weight decay는 바이어스와 LayerNorm/BatchNorm에는 적용하지 않는 것이 일반적이므로 파라미터 그룹을 나눠 설정하는 것이 좋습니다.

## 관련 노트
- [[Training/Training Loop Patterns]]
- [[Training/Loss Functions]]
- [[Advanced/AMP and Quantization]]
