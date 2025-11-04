## TL;DR
- 손실 함수는 모델 출력과 목표값의 차이를 수치화해 학습 방향을 결정합니다.
- 분류, 회귀, 순서 예측 등 문제 유형에 따라 적절한 손실을 선택해야 성능과 안정성이 확보됩니다.
- 메트릭과 손실의 설정이 불일치하면 학습이 원하는 방향으로 진행되지 않을 수 있습니다.

## 언제 쓰나
- 새로운 문제 유형에 맞는 손실을 선택하거나 조합해야 할 때
- 커스텀 손실을 구현해 모델 특성에 맞춰 세밀한 학습을 유도할 때
- 클래스 불균형, 레이블 노이즈 등 특이 케이스를 다룰 때

## 대표 손실 함수
|범주|함수|설명|사용 시점|
|---|---|---|---|
|분류|`nn.CrossEntropyLoss`|Softmax + NLL 결합|다중 클래스 분류|
|분류|`nn.BCEWithLogitsLoss`|시그모이드 + BCE 결합|멀티라벨 분류|
|회귀|`nn.MSELoss`|평균제곱오차|연속 값 회귀|
|회귀|`nn.L1Loss`|절대오차|노이즈에 강인한 회귀|
|랭킹|`nn.CosineEmbeddingLoss`|코사인 유사도 기반|임베딩 학습|
|세그먼테이션|`nn.NLLLoss2d`, Dice Loss(커스텀)|픽셀 단위 분류|불균형 클래스|
|특수|`nn.CTCLoss`|가변 길이 정렬|음성/문자 인식|

## 실습 예제
```python
import torch
import torch.nn as nn

logits = torch.randn(8, 10)
targets = torch.randint(0, 10, (8,))

ce_loss = nn.CrossEntropyLoss()(logits, targets)

multi_logits = torch.randn(8, 5)
multi_targets = torch.randint(0, 2, (8, 5)).float()
bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5, 1, 1, 1, 1]))(
    multi_logits, multi_targets
)

pred = torch.randn(8, 1)
reg_targets = torch.randn(8, 1)
mse_loss = nn.MSELoss()(pred, reg_targets)
```

## 커스텀 손실 만들기
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```

## 실수 주의
- `CrossEntropyLoss`는 라벨이 정수형 인덱스여야 하며 원-핫 인코딩을 그대로 넣으면 오류가 발생합니다.
- `BCEWithLogitsLoss`는 이미 시그모이드를 포함하고 있으므로 모델 출력에 시그모이드를 중복 적용하면 gradient가 사라집니다.
- 클래스 불균형 상황에서 `pos_weight` 등 가중치를 설정하지 않으면 학습이 한쪽으로 쏠릴 수 있습니다.
- 커스텀 손실에서 `reduction` 옵션을 설정하지 않으면 기본값이 `mean`이므로 합계를 의도할 경우 직접 처리해야 합니다.

## 관련 노트
- [[Training/Optimizers and Schedulers]]
- [[Training/Training Loop Patterns]]
- [[Evaluation/Metrics]]
