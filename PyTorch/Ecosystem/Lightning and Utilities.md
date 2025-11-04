## TL;DR
- PyTorch Lightning, Hydra, FastAI 등 유틸리티 프레임워크는 반복 코드를 줄이고 실험 관리를 돕습니다.
- Lightning은 학습 루프를 구조화하고, Hydra는 설정 관리, FastAI는 고수준 API를 제공합니다.
- 필요에 따라 순수 PyTorch와 혼용하거나, 프로젝트 요구에 맞춰 선택적으로 도입합니다.

## 언제 쓰나
- 복잡한 학습 루프/분산 설정을 자동화하고 싶을 때
- 여러 실험 설정을 체계적으로 관리하고자 할 때
- 빠른 프로토타이핑과 재사용이 중요한 프로젝트에서

## 주요 도구 비교
|도구|특징|장점|주의점|
|---|---|---|---|
|PyTorch Lightning|`LightningModule`로 학습 구조화|분산, 로깅 자동화|추상화 이해 필요|
|Hydra|구성 파일(.yaml) 기반 설정|실험 조합 관리|디렉터리 구조 주의|
|FastAI|고수준 API, 레시피 제공|빠른 프로토타입|추상화가 깊어 커스터마이징 난도|
|TorchMetrics|메트릭 모듈화|Lightning과 연동|단독 사용 가능|

## Lightning 예제
```python
import pytorch_lightning as pl

class LitClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
```

## Hydra 예제
```yaml
# configs/train.yaml
defaults:
  - model: resnet18
  - optimizer: adamw

trainer:
  max_epochs: 20
  gpus: 1
```
```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    print(cfg.trainer.max_epochs)

if __name__ == "__main__":
    main()
```

## 실수 주의
- Lightning과 순수 PyTorch를 혼합할 때는 학습 루프가 중복되지 않도록 주의하고, 콜백/로거가 중복 실행되지 않는지 확인하세요.
- Hydra는 작업 디렉터리를 변경하므로, 상대 경로 대신 `hydra.utils.to_absolute_path`를 사용하세요.
- 외부 유틸리티를 도입하면 디버깅이 어려울 수 있으니, 프로젝트 초기에 추상화 수준을 결정하고 팀원과 합의하세요.

## 관련 노트
- [[Training/Training Loop Patterns]]
- [[Evaluation/Logging and Visualization]]
- [[Projects/Project Template]]
