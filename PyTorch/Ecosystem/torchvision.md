## TL;DR
- `torchvision`은 이미지 데이터셋, 모델, 변환 기능을 제공하는 PyTorch 공식 서브 패키지입니다.
- `torchvision.datasets`, `torchvision.models`, `torchvision.transforms` 세 축을 이해하면 대부분의 컴퓨터 비전 워크플로를 구성할 수 있습니다.
- 사전학습 모델을 활용할 때는 `weights` 옵션과 입력 정규화를 정확히 맞춰야 합니다.

## 언제 쓰나
- CIFAR-10, ImageNet 등 표준 데이터셋으로 실험하고 싶을 때
- ResNet, EfficientNet 등 사전학습 모델을 로드해 전이학습을 할 때
- 이미지 증강과 데이터 전처리를 간단히 구성하고자 할 때

## 핵심 구성
|모듈|설명|예시|관련 노트|
|---|---|---|---|
|`torchvision.datasets`|표준 데이터셋 로더|`CIFAR10`, `ImageFolder`|[[Data/Dataset and DataLoader]]|
|`torchvision.models`|사전학습 모델|`resnet50(weights=...)`|[[Modeling/Core Layers]]|
|`torchvision.transforms`|이미지 변환/증강|`transforms.Compose([...])`|[[Data/Transforms/Transforms Overview]]|
|`torchvision.io`|이미지/비디오 I/O|`read_image`, `write_video`|[[Data/IO and Storage]]|

## 사전학습 모델 활용
```python
import torch
import torchvision.models as models
from torchvision import transforms

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

image = read_image("cat.jpg")  # [C, H, W]
batch = preprocess(image).unsqueeze(0)

with torch.no_grad():
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    top1 = probs.argmax(dim=1)
```

## 데이터셋 사용 예
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_set = datasets.CIFAR10(root="data", train=True,
                             download=True, transform=transform)
test_set = datasets.CIFAR10(root="data", train=False,
                            download=True, transform=transform)
```

## 실수 주의
- 사전학습 모델은 입력 해상도, 정규화(mean/std 값)가 명시되어 있으므로 `weights.transforms()`를 사용하거나 동일한 값을 수동으로 적용하세요.
- `ImageFolder`는 디렉터리 구조가 `class_name/이미지` 형태여야 하며, 잘못 구성하면 레이블이 꼬입니다.
- 데이터셋 다운로드 시 네트워크 연결 문제를 대비해 캐시 경로(`root`)를 명확히 지정하세요.
- 모델 파라미터 업데이트를 제한하려면 `for param in model.parameters(): param.requires_grad = False`를 잊지 마세요.

## 관련 노트
- [[Data/Transforms/Transforms Overview]]
- [[Projects/Image Classification Pipeline]]
- [[Deployment/Serving Options]]
