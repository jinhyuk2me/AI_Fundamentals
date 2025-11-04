## TL;DR
- Residual Block은 입력을 출력에 더해 gradient 소실을 완화하고 더 깊은 네트워크 학습을 가능하게 합니다.
- 3×3 합성곱 두 개와 identity shortcut으로 구성된 기본형은 ResNet-18/34 등에서 사용됩니다.
- Bottleneck 변형과 조합해 다양한 해상도·채널 조정 시나리오에 대응할 수 있습니다.

## 언제 쓰나
- 20층 이상의 CNN을 학습할 때 학습 난이도로 성능이 떨어지는 현상을 방지하고 싶을 때
- Skip connection을 추가해 MLP/CNN 등에서 학습 안정성과 수렴 속도를 높이고 싶을 때
- ResNet 기반 모델(Detection, Segmentation, Transformer 변형 등)을 커스터마이징할 때

## 핵심 아이디어
- **Identity Shortcut**: 입력을 그대로 더해 gradient가 직접 전달되도록 유지.
- **Normalization + Activation**: Conv → BatchNorm → ReLU 패턴으로 학습을 안정화.
- **Shape 정합**: 해상도나 채널이 변경될 경우 1×1 conv로 shortcut 경로를 조정.
- **Bottleneck(선택)**: 깊은 네트워크에서는 1×1-3×3-1×1로 연산량을 줄임.

## 구조 스케치
```
Input
 ├─ Conv 3×3 (in → hidden) → BatchNorm → ReLU
 ├─ Conv 3×3 (hidden → in) → BatchNorm
 └─ (선택) 다운샘플/채널조정 Conv 1×1
Add (Input + F(Input))
ReLU
```

## 실습 예제
```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, stride=1):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == hidden_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(in_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        identity = self.shortcut(x)
        return self.activation(out + identity)
```
- `ResidualBlock(64)`처럼 stride=1인 기본 설정은 해상도를 유지하며, `ResidualBlock(128, hidden_channels=128, stride=2)`는 해상도를 절반으로 줄입니다.

## 실수 주의
- shortcut 경로와 본 경로의 채널 수/stride가 다르면 shape mismatch가 발생하므로 반드시 downsample를 맞춰야 합니다.
- BatchNorm을 사용할 때 `model.train()`/`model.eval()` 모드를 혼동하면 추론 성능이 크게 저하될 수 있습니다.
- 아주 작은 batch size에서는 BatchNorm 대신 GroupNorm/LayerNorm 등을 고려해야 합니다.

## 관련 노트
- He et al., 2015, "Deep Residual Learning for Image Recognition"
- [[Modeling/Core Layers]]
- [[Modeling/Custom Blocks]]
- [[Projects/Image Classification Pipeline]]
