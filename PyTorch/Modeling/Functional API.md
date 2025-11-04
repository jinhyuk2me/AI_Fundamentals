## TL;DR
- `torch.nn.functional`은 상태를 가지지 않는 함수형 연산 모음으로, 레이어 클래스와 동일한 핵심 기능을 제공합니다.
- 커스텀 레이어 구현, 단발성 연산, 실험적 구조 테스트에 유용합니다.
- 파라미터가 필요한 경우 직접 `nn.Parameter`를 정의해 함께 사용해야 합니다.

## 언제 쓰나
- `forward`에서 단순 연산을 빠르게 적용하고 싶을 때 (예: `F.relu`)
- 상태 없는 연산을 조합해 커스텀 모듈을 만들 때
- 손실 함수를 클래스 대신 함수 형태로 호출할 때

## 대표 함수 분류
|분류|예시|설명|주의점|
|---|---|---|---|
|활성 함수|`F.relu`, `F.gelu`, `F.silu`|입력 텐서를 비선형 변환|`inplace=True` 사용 시 원본 변형|
|합성곱/풀링|`F.conv2d`, `F.max_pool2d`|입력 텐서와 가중치를 직접 전달|가중치는 파라미터로 관리해야 함|
|정규화|`F.batch_norm`, `F.layer_norm`|평균·분산 기반 정규화 수행|런닝 스탯을 직접 지정해야 함|
|드롭아웃|`F.dropout`, `F.dropout2d`|랜덤 비활성화|`training` 플래그 필요|
|손실 함수|`F.cross_entropy`, `F.mse_loss`|다양한 학습 손실 계산|인자 순서, reduction 옵션 주의|

## 실습 예제
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.kaiming_normal_(weight)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = F.conv2d(x, self.weight, bias=self.bias, padding=1)
        x = F.relu(x, inplace=True)
        return F.adaptive_avg_pool2d(x, 1)

module = FunctionalConv(3, 16, kernel_size=3)
dummy = torch.randn(4, 3, 32, 32)
print(module(dummy).shape)
```

## 실수 주의
- `F.dropout(x, training=self.training)`처럼 현재 모드에 따라 `training` 인자를 넘겨야 합니다.
- 정규화 함수(`F.batch_norm`)는 런닝 통계를 직접 넘겨야 하므로 일반적으로 클래스 버전(`nn.BatchNorm2d`)을 권장합니다.
- 파라미터를 직접 만들지 않으면 연산에 사용된 텐서가 학습 대상이 아니므로, `nn.Parameter`로 등록해야 합니다.

## 관련 노트
- [[Modeling/Module Basics]]
- [[Modeling/Core Layers]]
- [[Training/Loss Functions]]
