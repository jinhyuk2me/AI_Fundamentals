## TL;DR
- 1×1-3×3-1×1 합성곱으로 채널을 압축·복원해 깊은 네트워크의 연산량을 크게 줄이면서 성능을 유지합니다.
- Residual skip connection 덕분에 50층 이상에서도 gradient 소실 없이 학습이 가능합니다.
- ResNet-50/101/152, ResNeXt 등 대규모 비전 모델에서 사실상 표준 블록입니다.

## 언제 쓰나
- ImageNet 이상급 데이터셋에서 50층 이상의 ResNet/ResNeXt 구조를 구현할 때
- 3×3 합성곱을 많이 사용하지만 FLOPs/파라미터 수를 줄이고 싶을 때
- Transfer learning 시 pretrained ResNet 계열 가중치를 재사용하는 경우

## 핵심 아이디어
- **1×1 → 3×3 → 1×1**: 첫 1×1로 채널을 `mid_channels`까지 압축, 3×3으로 공간 특징 추출, 마지막 1×1로 채널 복원.
- **Residual Connection**: 입력을 출력에 더해 gradient 경로를 짧게 유지.
- **BatchNorm + ReLU**: 각 합성곱 뒤 정규화와 비선형성으로 안정성 확보.
- **Downsample 옵션**: stride나 채널 수가 달라질 때 shortcut 경로에서 shape을 맞춤.

## 구조 스케치
```
Input
 ├─ Conv 1×1 (C → C/r) → BN → ReLU
 ├─ Conv 3×3 (C/r → C/r) → BN → ReLU
 ├─ Conv 1×1 (C/r → C·expansion) → BN
 └─ (선택) Downsample Conv 1×1 (stride, channel match)
Add
ReLU
```

## 실습 예제
```python
import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)
```
- 예시: `BottleneckBlock(256, 64, stride=2, downsample=nn.Sequential(nn.Conv2d(256, 256, 1, stride=2, bias=False), nn.BatchNorm2d(256)))` 처럼 stride를 지정하면 출력 해상도를 절반으로 줄일 수 있습니다.

## 실수 주의
- `mid_channels`를 너무 작게 잡으면 정보 병목이 일어나고, 너무 크게 잡으면 FLOPs 절감 효과가 사라집니다.
- Downsample 경로와 본 경로의 stride/채널 수를 일치시키지 않으면 텐서 shape이 맞지 않아 오류가 납니다.
- BatchNorm은 학습/추론 모드에 따라 동작이 달라지므로 `model.eval()` 호출 여부에 주의하세요.

## 관련 노트
- He et al., 2015/2016, "Deep Residual Learning for Image Recognition"
- Xie et al., 2017, "Aggregated Residual Transformations for Deep Neural Networks"
- [[Modeling/Custom Blocks]]
- [[Modeling/Design Notes/Residual Block]]
