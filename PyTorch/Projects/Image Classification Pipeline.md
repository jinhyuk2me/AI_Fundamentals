## TL;DR
- 이미지 분류 파이프라인은 데이터 로딩, 증강, 모델 구성, 학습, 평가, 배포 단계로 이어집니다.
- 기본 베이스라인으로 ResNet/EfficientNet과 CrossEntropyLoss, AdamW를 사용해 빠르게 성능을 확인합니다.
- 실험 로그, 체크포인트, 하이퍼파라미터를 기록해 비교 가능한 실험 환경을 유지합니다.

## 언제 쓰나
- 이미지 분류 문제를 처음 해결하거나 신규 데이터셋을 적용할 때
- 전이학습(Transfer Learning)으로 빠르게 성능을 확보하고자 할 때
- 이후 세부 개선(세분화, 다중 레이블 등)을 위한 기반 파이프라인이 필요할 때

## 단계별 구성
|단계|설명|주요 작업|관련 노트|
|---|---|---|---|
|데이터 준비|이미지 로딩, 증강, split|`ImageFolder`, 커스텀 Dataset|[[Data/Dataset and DataLoader]], [[Data/Transforms/Transforms Overview]]|
|모델|사전학습 백본 + 분류 헤드|`torchvision.models` 활용|[[Modeling/Core Layers]], [[torchvision]]|
|학습|Optimizer, Scheduler, AMP|AdamW + Cosine LR|[[Training/Optimizers and Schedulers]], [[Advanced/AMP and Quantization]]|
|평가|Accuracy, F1, Confusion Matrix|TensorBoard/W&B 로깅|[[Evaluation/Metrics]], [[Evaluation/Logging and Visualization]]|
|배포|TorchScript 또는 ONNX Export|FastAPI/TorchServe|[[Deployment/Serving Options]]|

## 예시 코드 스니펫
```python
from torchvision import datasets, transforms, models
import torch.nn as nn

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

train_ds = datasets.ImageFolder("data/train", transform=transform_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

weights = models.ResNet18_Weights.DEFAULT
backbone = models.resnet18(weights=weights)
num_features = backbone.fc.in_features
backbone.fc = nn.Linear(num_features, num_classes)

model = backbone.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

## 체크포인트 전략
- `best.pt`: 최고 검증 정확도 모델
- `last.pt`: 마지막 에폭용 롤백 모델
- `metrics.json`: 에폭별 손실, 정확도 기록
- `config.yaml`: 데이터 경로, 하이퍼파라미터

## 실수 주의
- 클래스 불균형이 심하면 데이터 증강뿐 아니라 `WeightedRandomSampler`나 `class_weight` 손실을 고려하세요.
- 전이학습 시 초기에는 Backbone 파라미터 고정을 고려하고, 괜찮으면 전체 파인튜닝으로 전환하세요.
- 이미지 정규화(mean/std)가 사전학습 모델과 일치하는지 확인하세요.
- 배포 시 TorchScript 변환이 안 되는 연산이 있는지 미리 테스트하세요.

## 관련 노트
- [[Tensor 변환]]
- [[torchvision]]
- [[Deployment/Serving Options]]
