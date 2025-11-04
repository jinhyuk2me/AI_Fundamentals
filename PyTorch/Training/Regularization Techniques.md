## TL;DR
- 정규화는 모델의 과적합을 완화하고 일반화 성능을 높이는 전략입니다.
- Weight decay, Dropout, 데이터 증강, label smoothing, early stopping 등을 상황에 맞게 조합합니다.
- 학습 루프와 모델 정의 단계 모두에서 정규화 요소를 배치해야 효과가 극대화됩니다.

## 언제 쓰나
- 훈련 데이터 정확도는 높지만 검증/테스트 성능이 낮을 때
- 소량 데이터나 라벨 노이즈가 심한 환경에서 학습할 때
- 대규모 모델을 안정적으로 학습시키고 싶을 때

## 주요 기법
|기법|설명|적용 위치|관련 노트|
|---|---|---|---|
|Weight Decay (L2)|파라미터 크기 제한|Optimizer(`weight_decay`)|[[Training/Optimizers and Schedulers]]|
|Dropout|활성 일부 무작위 제거|모델 레이어|[[Modeling/Core Layers]]|
|데이터 증강|입력 변형으로 데이터 확대|Transforms/Dataloader|[[Data/Transforms/Transforms Overview]]|
|Label Smoothing|레이블에 여유 값 추가|손실 함수|[[Training/Loss Functions]]|
|Early Stopping|검증 성능 하락 시 중단|학습 루프|[[Evaluation/Validation and Early Stopping]]|
|Gradient Clipping|Gradient 폭주 방지|학습 루프|[[Training/Training Loop Patterns]]|

## 실습 예제
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-3,
                              weight_decay=1e-2)

def label_smoothing(targets, num_classes, smoothing=0.1):
    off_value = smoothing / (num_classes - 1)
    on_value = 1.0 - smoothing
    smoothed = torch.full((targets.size(0), num_classes), off_value,
                          device=targets.device)
    smoothed.scatter_(1, targets.unsqueeze(1), on_value)
    return smoothed
```

## 실수 주의
- Weight decay는 Bias, BatchNorm, LayerNorm에는 적용하지 않는 것이 일반적이므로 파라미터 그룹을 분리하는 것이 좋습니다.
- Dropout 확률을 너무 높게 설정하면 학습이 수렴하지 않을 수 있습니다. 보통 0.1~0.5 범위를 먼저 테스트합니다.
- Label smoothing은 평가 시 정확도 계산과 호환되는지 확인해야 합니다. 원-핫 라벨이 필요하다면 따로 변환해야 합니다.
- Early stopping은 검증 데이터가 충분히 대표성을 갖는지 확인해야 하며, patience 값을 너무 작게 잡으면 수렴 전에 중단될 수 있습니다.

## 관련 노트
- [[Training/Loss Functions]]
- [[Evaluation/Validation and Early Stopping]]
- [[Advanced/Profiling and Debugging]]
