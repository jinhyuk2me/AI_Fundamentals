## TL;DR
- 메트릭은 모델 성능을 정량화하며, 문제 유형에 따라 적절한 지표를 선택해야 합니다.
- Classification/Regression/Ranking/Detection 등 각 과제별 전용 메트릭을 이해하면 실험 비교가 명확해집니다.
- 배치 단위로 업데이트하고 최종적으로 집계하는 패턴을 사용하면 메모리 사용을 줄일 수 있습니다.

## 언제 쓰나
- 학습 과정에서 성능을 모니터링하고 모델 선택 기준을 세울 때
- 실험 결과를 비교·재현 가능하게 만들고자 할 때
- 논문·보고서에서 요구하는 지표를 계산해야 할 때

## 대표 메트릭
|문제 유형|지표|설명|주의점|
|---|---|---|---|
|이진/다중 분류|Accuracy, Precision, Recall, F1|클래스 분류 정확도|클래스 불균형 시 Accuracy 주의|
|확률 기반|ROC-AUC, PR-AUC|확률 분포 비교|threshold 선택 필요|
|회귀|MAE, MSE, RMSE, R^2|연속 값 오차|스케일 민감|
|랭킹|MAP, NDCG|순위 품질|Cutoff 설정 필요|
|세그먼트/검출|mIoU, mAP|픽셀/박스 수준 정확도|기준 IoU, score threshold|

## 실습 예제
```python
import torch
from sklearn.metrics import accuracy_score, f1_score

logits = torch.randn(32, 5)
targets = torch.randint(0, 5, (32,))

pred = logits.argmax(dim=1).cpu().numpy()
true = targets.cpu().numpy()

acc = accuracy_score(true, pred)
macro_f1 = f1_score(true, pred, average="macro")
print(acc, macro_f1)
```

### 누적 업데이트 패턴
```python
class MetricTracker:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, targets):
        preds = logits.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self):
        return self.correct / max(self.total, 1)
```

## 실수 주의
- 배치 단위로 accuracy를 계산해 평균을 취하면 가중치가 다르게 적용될 수 있으므로 전체 카운트를 누적하거나 `torchmetrics` 같은 라이브러리를 활용하세요.
- ROC-AUC 계산에는 클래스별 확률이 필요하므로 `softmax` 후 값을 사용하거나 `sklearn.metrics`의 입력 요구사항을 확인하세요.
- mAP, mIoU 등은 구현이 복잡하므로 검증된 라이브러리를 사용하는 것이 안전합니다.
- 훈련 데이터로 메트릭을 과도하게 최적화하면 평가 지표와 실제 성능이 괴리될 수 있습니다.

## 관련 노트
- [[Evaluation/Validation and Early Stopping]]
- [[Evaluation/Logging and Visualization]]
- [[Projects/Project Template]]
