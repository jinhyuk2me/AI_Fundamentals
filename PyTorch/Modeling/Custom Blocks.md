## TL;DR
- 커스텀 블록은 기존 레이어를 결합해 재사용 가능한 하위 모듈을 만드는 패턴입니다.
- Residual, Inception, Attention, Multi-Branch 구조 등을 이해하면 대규모 모델 설계가 쉬워집니다.
- `nn.Module` 조합, skip connection 구현, 파라미터 공유 등을 정확히 다뤄야 합니다.

## 언제 쓰나
- 기존 백본을 확장하거나 새로운 아키텍처를 연구할 때
- 복잡한 연산을 캡슐화해 가독성과 재사용성을 높이고 싶을 때
- 실험을 빠르게 반복하면서 구조적인 변화를 적용할 때

## 대표 블록 패턴
|블록|설명|핵심 구성|관련 노트|
|---|---|---|---|
|[[Modeling/Design Notes/Residual Block\|Residual Block]]|입출력을 더해 gradient 흐름 개선|Conv → BN → ReLU → Conv + Skip|[[Tensor 가속]]|
|[[Modeling/Design Notes/Bottleneck Residual Block\|Bottleneck Block]]|채널 축소후 확장|1×1 → 3×3 → 1×1 구조|[[Modeling/Core Layers]]|
|[[Modeling/Design Notes/Inception Block\|Inception Block]]|다중 커널 병렬 적용|1×1, 3×3, 5×5, Pool concat|[[Modeling/Functional API]]|
|[[Modeling/Design Notes/Squeeze-Excitation Block\|Squeeze-Excitation]]|채널 주의로 강조/억제|Global Pool → FC → Sigmoid|[[Tensor 연산]]|
|[[Modeling/Design Notes/Transformer Encoder Block\|Transformer Encoder]]|Self-Attention 기반 시퀀스 처리|MHA + FFN + Residual/Norm|[[Projects/NLP Pipeline]]|

## Residual Block 예제
```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)
```

## Attention Block 예제 (Squeeze-Excitation)
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
```

## 실수 주의
- Skip connection을 사용할 때 input과 output 채널 수가 다르면 다운샘플링/1×1 Convolution 등으로 맞춰야 합니다.
- 병렬 분기(Inception 등)를 사용할 때 출력 텐서의 shape이 일치하도록 패딩/stride를 조정하세요.
- Attention/SE 블록은 배치 크기가 작을 때 불안정할 수 있으므로 정상적으로 작동하는지 모니터링하세요.
- 커스텀 블록 수가 많아지면 메모리 사용량과 연산량이 급증하므로 프로파일링을 병행하세요.

## 관련 노트
- [[Modeling/Module Basics]]
- [[Modeling/Core Layers]]
- [[Projects/Image Classification Pipeline]]
- [[Modeling/Design Notes/Neural Network Modeling Roadmap]]
- [[Modeling/Design Notes/Bottleneck Residual Block]]
- [[Modeling/Design Notes/Inception Block]]
- [[Modeling/Design Notes/Squeeze-Excitation Block]]
- [[Modeling/Design Notes/Transformer Encoder Block]]
