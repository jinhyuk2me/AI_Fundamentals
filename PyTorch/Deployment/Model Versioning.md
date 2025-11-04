## TL;DR
- 모델 버전 관리는 모델 아티팩트, 코드, 데이터, 하이퍼파라미터를 함께 추적해 재현성과 롤백을 보장합니다.
- 버전 명명 규칙, 저장소 구조, 메타데이터(지표, 환경 정보)를 일관되게 유지해야 합니다.
- MLflow, DVC, Weights & Biases Artifacts, Model Registry 등을 활용하면 자동화할 수 있습니다.

## 언제 쓰나
- 주기적으로 모델을 업데이트하거나, 여러 모델을 동시에 운영해야 할 때
- 실험 결과를 장기간 보존하고 재학습 또는 롤백이 필요한 환경에서
- 규제나 컴플라이언스 요구사항으로 모델 추적이 필요한 경우

## 핵심 전략
|전략|설명|예시|관련 노트|
|---|---|---|---|
|버전 명명 규칙|의미 있는 버전 태깅|`model_vYYYYMMDD_HHMM`|[[Projects/Project Template]]|
|메타데이터 기록|지표, 데이터셋, 하이퍼파라미터 저장|`metrics.json`, `params.yaml`|[[Evaluation/Logging and Visualization]]|
|아티팩트 저장소|모델 파일 보관|S3, GCS, MinIO, MLflow|[[Deployment/Serving Options]]|
|모델 레지스트리|승인/배포 단계 관리|MLflow Model Registry|[[Project Tooling]]|

## MLflow 예제
```python
import mlflow

mlflow.set_experiment("pytorch-classifier")

with mlflow.start_run(run_name="resnet50_v1"):
    mlflow.log_params({"lr": 1e-3, "batch_size": 64})
    mlflow.log_metrics({"val_acc": 0.87, "val_loss": 0.45})
    mlflow.pytorch.log_model(model, "model")
```

## DVC 예제
```bash
dvc init
dvc add models/best.pt
git add models/best.pt.dvc .gitignore
git commit -m "Track best model"
dvc push
```

## 실수 주의
- 모델 파일과 코드를 따로 관리하면 재현성이 떨어집니다. `requirements.txt` 혹은 `poetry.lock` 등 환경 정보도 함께 버전 관리하세요.
- 대용량 파일을 Git에 직접 커밋하면 저장소가 비대해지므로 DVC, Git LFS 등을 사용하세요.
- 레지스트리/아티팩트 저장소 접근 권한을 관리하고, 누가 어떤 모델을 배포했는지 로그를 남기세요.
- 배포된 모델이 교체될 때 지표가 악화되지 않았는지 모니터링하고, 필요 시 즉시 롤백 가능한 체계를 마련하세요.

## 관련 노트
- [[Deployment/Serving Options]]
- [[Evaluation/Logging and Visualization]]
- [[Projects/Project Template]]
