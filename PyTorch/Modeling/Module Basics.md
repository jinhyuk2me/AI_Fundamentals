## TL;DR
- `nn.Module`은 PyTorch 모델의 기본 단위로, 파라미터와 하위 모듈을 자동으로 관리해 줍니다.
- `forward` 정의, `__init__`에서 하위 모듈 등록, `parameters()`/`named_parameters()` 활용이 핵심 패턴입니다.
- 트레이닝·추론 전환은 `model.train()` / `model.eval()`로 제어하며, `state_dict`로 저장·복원합니다.

## 언제 쓰나
- 새로운 모델 아키텍처를 설계할 때 기본 틀을 구현할 때
- 사전학습 모델을 확장하거나 일부 레이어를 교체할 때
- 서브모듈을 조합한 블록을 재사용 가능한 형태로 만들 때

## 주요 개념
| 개념             | 설명                                          | 기억 포인트                        |
| -------------- | ------------------------------------------- | ----------------------------- |
| `nn.Module` 상속 | 모든 학습 가능한 모듈의 베이스 클래스                       | `__init__`에서 부모 호출 필수         |
| 하위 모듈 등록       | `self.layer = nn.Linear(...)`처럼 속성에 할당      | 자동으로 `parameters()`에 포함       |
| `forward` 메서드  | 순전파 로직 정의                                   | `__call__`이 내부에서 `forward` 호출 |
| 파라미터 접근        | `model.parameters()` / `named_parameters()` | 학습/로깅/초기화에 사용                 |
| 상태 저장          | `model.state_dict()`와 `load_state_dict()`   | 텐서만 저장되므로 구조 동일해야 함           |
| 모드 전환          | `model.train()`, `model.eval()`             | Dropout, BatchNorm 등 동작 변경    |

## 실습 예제
```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        return self.activation(out + x)

model = ResidualBlock(64, 128)
model.train()

dummy = torch.randn(8, 64, 32, 32)
out = model(dummy)

print(out.shape)
print(len(list(model.parameters())))
```

## 실수 주의
- `__init__`에서 `super().__init__()`를 호출하지 않으면 파라미터가 등록되지 않습니다.
- `forward` 외에 순전파에 필요한 추가 메서드를 정의했다면 `__call__`이 아닌 별도 호출이 필요합니다.
- `model.eval()`을 잊으면 추론 중에도 Dropout 등이 활성화되어 결과가 불안정해질 수 있습니다.
- `state_dict` 로드 시 키가 일치하지 않으면 오류가 발생하므로 모듈 구조 변경 후에는 새롭게 저장해야 합니다.

## 관련 노트
- [[Tensor Autograd]]
- [[Modeling/Core Layers]]
- [[Training/Training Loop Patterns]]
- [[Modeling/Design Notes/Neural Network Modeling Roadmap]]
- [[Modeling/Design Notes/Residual Block]]
