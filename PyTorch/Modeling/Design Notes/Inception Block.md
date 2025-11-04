## TL;DR
- Inception Block은 1×1, 3×3, 5×5, Pool 분기를 동시에 적용해 멀티 스케일 특징을 포착합니다.
- 1×1 차원 축소로 연산량을 억제하면서도 넓은 receptive field를 확보할 수 있습니다.
- GoogLeNet 계열에서 검증된 구조로, 폭을 넓히는 전략이 필요한 경우 유용합니다.

## 언제 쓰나
- 다양한 크기의 패턴이 섞여 있는 이미지에서 여러 스케일을 한 번에 처리하고 싶을 때
- ResNet 스타일 대신 병렬 경로로 모델 표현력을 확장하려는 경우
- Inception v1~v4, Inception-ResNet, EfficientNet 등 멀티 브랜치 기반 네트워크를 구현할 때

## 핵심 아이디어
- **병렬 분기**: 1×1, 3×3, 5×5(또는 3×3 dilated), MaxPool 경로를 동시에 수행.
- **1×1 차원 축소**: 큰 커널 앞에 1×1 conv를 두어 채널을 줄이고 FLOPs를 절약.
- **Concatenation**: 분기 결과를 채널 방향으로 concat하여 다음 레이어 입력으로 사용.
- **Norm + Activation**: 각 분기마다 BatchNorm과 ReLU를 넣어 안정적인 학습을 보장.

## 구조 스케치
```
Input
 ├─ 1×1 Conv
 ├─ 1×1 Conv → 3×3 Conv
 ├─ 1×1 Conv → 5×5 Conv (또는 3×3 dilated)
 ├─ 3×3 MaxPool → 1×1 Conv
Concat (채널 방향)
```

## 실습 예제
```python
import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels,
                 ch1x1,
                 ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5,
                 pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x),
                   self.branch3(x), self.branch4(x)]
        return torch.cat(outputs, dim=1)
```
- Inception v1 기준으로 `InceptionBlock(192, 64, 96, 128, 16, 32, 32)` 같은 채널 구성이 사용됩니다.

## 실수 주의
- 분기별 출력 채널 수 합이 지나치게 크면 GPU 메모리 사용량이 급증하므로 합계를 제한하세요.
- 작은 입력 해상도에서는 5×5 커널 대신 3×3 두 번으로 대체해 aliasing을 줄이는 편이 좋습니다.
- MaxPool 분기에는 stride=1을 두고 padding으로 해상도를 맞추는 것이 일반적입니다.

## 관련 노트
- Szegedy et al., 2015, "Going Deeper with Convolutions"
- Szegedy et al., 2016, "Rethinking the Inception Architecture for Computer Vision"
- [[Modeling/Custom Blocks]]
- [[Projects/Image Classification Pipeline]]
