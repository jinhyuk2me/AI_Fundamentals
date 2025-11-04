## TL;DR
- 핵심 레이어는 `Linear`, `Conv`, `RNN`, `Normalization`, `Dropout` 등 범용적으로 쓰이는 모듈들로 구성됩니다.
- 입력/출력 형태, 파라미터 수, 연산 특성을 이해하면 모델 설계와 디버깅이 수월해집니다.
- 대부분의 레이어는 `bias`, `inplace`, `padding` 같은 옵션으로 미세 조정이 가능합니다.

## 언제 쓰나
- 이미지·시퀀스·표 형태 데이터를 처리하는 기본 모델을 만들 때
- 아키텍처를 커스터마이징하면서 기존 레이어의 동작을 확실히 이해해야 할 때
- 파라미터 수, 연산량을 계산해 모델 크기를 조절할 때

## 주요 레이어 분류

### 선형 계열
|레이어|설명|입력/출력|주요 옵션|
|---|---|---|---|
|`nn.Linear`|완전연결 레이어|`(N, in_features)` → `(N, out_features)`|`bias=True`, `dtype`, `device`|
|`nn.Bilinear`|두 입력 결합|`(N, in1)`, `(N, in2)` → `(N, out)`|`bias`, `device`|

### 합성곱 계열
|레이어|설명|입력/출력|주요 옵션|
|---|---|---|---|
|`nn.Conv1d/2d/3d`|1/2/3차원 합성곱|`(N, C_in, L/HxW/HxWxD)`|`kernel_size`, `stride`, `padding`, `groups`|
|`nn.ConvTranspose1d/2d/3d`|전치 합성곱(업샘플링)|해당 차원 입력/출력|`output_padding`, `dilation`|
|`nn.MaxPool*`, `nn.AvgPool*`|값/평균 기반 풀링|차원 축소|`kernel_size`, `stride`|

### 순환 및 시퀀스
|레이어|설명|입력/출력|주요 옵션|
|---|---|---|---|
|`nn.RNN`|단순 순환|`(seq_len, batch, input_size)`|`num_layers`, `bidirectional`|
|`nn.GRU`, `nn.LSTM`|게이트 순환|동일|`hidden_size`, `dropout`|
|`nn.TransformerEncoderLayer`|Self-Attention 기반|`(seq_len, batch, d_model)`|`nhead`, `dim_feedforward`|

### 정규화
|레이어|설명|적용 축|주요 옵션|
|---|---|---|---|
|`nn.BatchNorm1d/2d/3d`|배치 단위 정규화|각 데이터 축|`momentum`, `eps`, `affine`|
|`nn.LayerNorm`|특성 축 정규화|마지막 차원|`normalized_shape`|
|`nn.GroupNorm`|채널 그룹화 정규화|채널 조각|`num_groups`|

### 정규화/드롭아웃
|레이어|설명|주요 특징|
|---|---|---|
|`nn.Dropout`, `nn.Dropout2d/3d`|랜덤으로 일부 활성 무시|`model.train()` 상태에서만 적용|
|`nn.AlphaDropout`, `nn.FeatureAlphaDropout`|Self-Normalizing 네트워크용|SELU 활성화와 호환|

## 실습 예제
```python
import torch
import torch.nn as nn

class MiniCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = MiniCNN()
dummy = torch.randn(8, 3, 32, 32)
print(model(dummy).shape)
```

## 실수 주의
- 합성곱의 입력/출력 차원 순서(`NCHW`)를 혼동하면 shape 불일치 오류가 발생합니다.
- `Dropout`은 `model.eval()` 상태에서 자동으로 비활성화되지만, 커스텀 레이어에선 동일 동작을 구현해야 합니다.
- RNN 계열 레이어는 입력 shape(`seq_len`, `batch`, `feature`)를 지켜야 하며, `batch_first=True` 옵션을 혼동하지 않도록 주의하세요.

## 관련 노트
- [[Modeling/Module Basics]]
- [[Modeling/Functional API]]
- [[Training/Regularization Techniques]]
- [[Modeling/Design Notes/Neural Network Modeling Roadmap]]
