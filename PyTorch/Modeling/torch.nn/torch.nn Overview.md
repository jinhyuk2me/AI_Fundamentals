## TL;DR
- `torch.nn` 패키지는 신경망 레이어, 활성 함수, 손실 함수 등 모듈화된 구성 요소를 제공합니다.
- 대부분의 클래스는 `nn.Module`을 상속하므로 조합과 재사용이 용이합니다.
- `nn.functional`은 동일 연산의 함수형 버전을 제공해 커스텀 레이어 구현에 활용할 수 있습니다.

## 언제 쓰나
- 모델을 구성할 때 표준 레이어와 손실 함수를 빠르게 조합하고 싶을 때
- 함수형 API와 조합해 커스텀 연산을 만들고자 할 때
- PyTorch 생태계의 선행 코드(예: torchvision 모델)를 분석·수정할 때

## 하위 모듈 개요
|서브패키지|내용|대표 클래스/함수|관련 노트|
|---|---|---|---|
|`nn.modules`|레이어, 컨테이너, 합성 모듈|`Linear`, `Conv2d`, `Sequential`|[[Modeling/Core Layers]]|
|`nn.functional`|함수형 레이어/활성/손실|`F.relu`, `F.dropout`|[[Modeling/Functional API]]|
|`nn.parameter`|파라미터 래퍼|`Parameter`, `ParameterList`|[[Modeling/Module Basics]]|
|`nn.init`|초기화 유틸|`xavier_uniform_`, `kaiming_normal_`|[[Training/Regularization Techniques]]|
|`nn.utils`|보조 기능|`clip_grad_norm_`, `weight_norm`|[[Training/Training Loop Patterns]]|

## 실습 예제
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = ConvClassifier()
inputs = torch.randn(16, 3, 32, 32)
logits = model(inputs)
loss = F.cross_entropy(logits, torch.randint(0, 10, (16,)))
```

## 실수 주의
- `nn.functional`은 상태를 저장하지 않으므로 BatchNorm, Dropout처럼 학습 상태가 필요한 연산은 클래스 버전을 사용해야 합니다.
- `nn.Sequential` 내부에서 스킵 연결 등 복잡한 흐름을 구현할 수 없으므로 별도 모듈로 작성해야 합니다.
- 초기화는 자동으로 이뤄지지만, 모델 특성에 따라 `nn.init`으로 명시적 초기화를 하는 것이 안정적일 수 있습니다.

## 관련 노트
- [[Modeling/Module Basics]]
- [[Modeling/Core Layers]]
- [[Training/Loss Functions]]
