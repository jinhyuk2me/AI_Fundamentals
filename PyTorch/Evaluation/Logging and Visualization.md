## TL;DR
- 학습 과정의 손실, 메트릭, 하이퍼파라미터를 기록하면 실험 재현성과 디버깅 효율이 크게 향상됩니다.
- TensorBoard, Weights & Biases(W&B), CSV Logger 등 다양한 도구를 활용할 수 있습니다.
- 시각화를 통해 과적합, 학습률 문제, 데이터 품질 이슈 등을 빠르게 발견할 수 있습니다.

## 언제 쓰나
- 여러 실험을 비교하거나 팀원과 결과를 공유해야 할 때
- 하이퍼파라미터 튜닝과 모델 선택 기준을 명확히 하고자 할 때
- 학습이 불안정하거나 성능이 갑자기 떨어지는 문제를 조사할 때

## 주요 도구
|도구|특징|사용 예|비고|
|---|---|---|---|
|TensorBoard|로컬/원격 웹 UI|`SummaryWriter`|PyTorch 내장 지원|
|Weights & Biases|클라우드 로그, 협업|`wandb.init()`|프로젝트 공유 용이|
|CSV Logger|간단한 파일 기록|`csv.writer`|의존성 최소화|
|Matplotlib/Seaborn|커스텀 시각화|학습 곡선, Confusion Matrix|보고서 작성에 유용|

## TensorBoard 예제
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/experiment1")

for epoch in range(1, 11):
    train_loss = ...
    val_loss = ...
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f"params/{name}", param, epoch)

writer.close()
```

## W&B 예제
```python
import wandb

wandb.init(project="pytorch-study", config={"lr": 1e-3, "batch_size": 64})

for epoch in range(1, 6):
    train_metrics = {"loss": 0.5, "accuracy": 0.8}
    val_metrics = {"loss": 0.6, "accuracy": 0.78}
    wandb.log({
        "train/loss": train_metrics["loss"],
        "train/acc": train_metrics["accuracy"],
        "val/loss": val_metrics["loss"],
        "val/acc": val_metrics["accuracy"],
        "epoch": epoch
    })

wandb.finish()
```

## 실수 주의
- 로그 파일이 무한히 커지는 것을 방지하려면 주기적으로 정리하거나 보관 주기를 설정하세요.
- GPU 환경에서 TensorBoard를 실행할 때는 `--load_fast=false` 옵션을 사용하면 메모리 사용량이 줄어듭니다.
- 외부 서비스를 사용할 때는 API 키와 프로젝트 이름을 정확히 설정하고, 민감한 데이터가 포함되지 않는지 확인하세요.
- 로그 시점을 통일하지 않으면 에폭별 비교가 어려워지므로, 학습과 검증 로그를 동일 축으로 기록하는 것이 좋습니다.

## 관련 노트
- [[Evaluation/Metrics]]
- [[Evaluation/Validation and Early Stopping]]
- [[Projects/Project Template]]
