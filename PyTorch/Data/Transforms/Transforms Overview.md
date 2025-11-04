## TL;DR
- 변환(transforms)은 원본 데이터를 텐서로 변환하거나 데이터 증강을 적용해 모델 학습을 돕습니다.
- `torchvision.transforms`는 이미지 중심 변환을 제공하고, 사용자 정의 `__call__` 클래스로 확장할 수 있습니다.
- 파이프라인은 `Compose`로 묶어 DataLoader에 적용하며, 배치 단위 전처리가 필요하면 `collate_fn`도 고려합니다.

## 언제 쓰나
- 이미지/오디오/텍스트 데이터를 PyTorch 텐서로 바꾸고 정규화할 때
- 학습 시 데이터 증강을 적용해 일반화 성능을 높이고 싶을 때
- 복잡한 전처리 단계를 모듈화해 재사용하고자 할 때

## 주요 구성 요소
|종류|예시|설명|관련 노트|
|---|---|---|---|
|형식 변환|`transforms.ToTensor()`, `transforms.Normalize`|PIL/NumPy → Tensor, 정규화|[[Tensor 변환]]|
|기하학 변환|`transforms.RandomHorizontalFlip`, `transforms.RandomCrop`|이미지 증강|[[Projects/Image Classification Pipeline]]|
|색상/강도|`transforms.ColorJitter`, `transforms.RandomGrayscale`|밝기/대비 변화|[[Training/Regularization Techniques]]|
|컴포즈|`transforms.Compose([...])`|순차적 변환 묶기|[[Data/Dataset and DataLoader]]|
|사용자 정의|`class MyTransform: def __call__(self, sample)`|특정 데이터 구조 변환|[[Projects/Project Template]]|

## 실습 예제
```python
from torchvision import transforms
from PIL import Image
import torch

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

image = Image.open("sample.jpg").convert("RGB")
tensor = transform(image)
print(tensor.shape, tensor.dtype)

class AddGaussianNoise:
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return tensor + noise

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.1),
])
```

## 실수 주의
- `ToTensor()`는 픽셀 값을 `[0, 1]` 범위로 스케일링하고 채널 순서를 `(C, H, W)`로 바꾸므로 모델 입력과 맞춰야 합니다.
- 변환은 보통 학습 데이터에만 적용하며, 검증/테스트에는 최소한의 정규화만 사용합니다.
- 사용자 정의 변환은 `__call__`에서 입력 타입(딕셔너리, 튜플 등)을 명확히 처리해야 합니다.
- 배치 단위로 변환이 필요하면 DataLoader의 `collate_fn`이나 모델의 전처리 모듈에서 수행하세요.

## 관련 노트
- [[Data/Dataset and DataLoader]]
- [[Training/Regularization Techniques]]
- [[Projects/Image Classification Pipeline]]
