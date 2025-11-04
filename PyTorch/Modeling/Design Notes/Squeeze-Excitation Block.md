## TL;DR
- Squeeze-Excitation(SE) 블록은 채널별 가중치를 학습해 중요한 채널은 강조, 덜 중요한 채널은 억제합니다.
- Global Average Pooling → MLP(감쇠/확장) → Sigmoid 게이트로 구성되며, 파라미터 증가가 매우 적습니다.
- ResNet, EfficientNet 등 다양한 CNN에 삽입해 손쉽게 정확도를 끌어올릴 수 있습니다.

## 언제 쓰나
- CNN 특징맵의 채널 간 상관관계를 학습해 성능을 향상시키고 싶을 때
- 기존 블록 구조를 크게 바꾸지 않고 attention 요소를 추가하려는 경우
- ImageNet, Detection, Segmentation 등에서 효율적인 채널 attention이 필요할 때

## 핵심 아이디어
- **Squeeze**: Global AvgPool로 각 채널의 전역 통계를 1×1 특징으로 축소.
- **Excitation**: 두 개의 FC(또는 1×1 Conv)로 채널 중요도 벡터를 학습 후 Sigmoid로 정규화.
- **Scale**: 학습된 채널 게이트를 원본 feature map에 곱해 강조/억제.
- **Reduction Ratio**: 중간 차원 `C/r`로 축소해 파라미터 수와 연산량을 제어.

## 구조 스케치
```
Input (B, C, H, W)
 ├─ Global AvgPool → (B, C, 1, 1)
 ├─ FC (C → C/r) → ReLU
 ├─ FC (C/r → C) → Sigmoid
 └─ Rescale: Output = Input × Gate
```

## 실습 예제
```python
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
```
- ResNet 블록에 삽입하려면 본 경로의 마지막 BatchNorm 뒤에 `x = se(x)` 형태로 연결하면 됩니다.

## 실수 주의
- reduction ratio `r`이 너무 작으면 파라미터 수가 급증하고, 너무 크면 attention 효과가 약해집니다.
- FP16 환경에서는 Sigmoid 출력의 숫자 안정성을 위해 `torch.sigmoid` 이후 clamping을 고려하세요.
- Bottleneck 블록과 함께 사용할 때는 SE 모듈을 마지막 conv 출력에만 적용해야 skip connection과 shape이 일치합니다.

## 관련 노트
- Hu et al., 2018, "Squeeze-and-Excitation Networks"
- Woo et al., 2018, "CBAM: Convolutional Block Attention Module"
- [[Modeling/Custom Blocks]]
- [[Projects/Image Classification Pipeline]]
