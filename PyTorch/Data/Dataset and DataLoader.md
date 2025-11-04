## TL;DR
- `Dataset`은 샘플을 정의하고, `DataLoader`는 배치 단위로 불러오는 반복자를 생성합니다.
- `__len__`과 `__getitem__`을 구현하면 커스텀 Dataset을 만들 수 있습니다.
- 배치 크기, 샘플링 전략, 멀티프로세싱 설정을 통해 입출력 병목을 줄입니다.

## 언제 쓰나
- 커스텀 데이터셋을 PyTorch 학습 루프에 연결하고 싶을 때
- 대용량 데이터를 효율적으로 배치 처리하거나 셔플링할 때
- 학습/검증/테스트 파이프라인을 일관되게 구성해야 할 때

## 주요 구성 요소
|구성|설명|핵심 메서드/옵션|관련 노트|
|---|---|---|---|
|`torch.utils.data.Dataset`|데이터셋 기본 인터페이스|`__len__`, `__getitem__`|[[Tensor 입출력]]|
|`torch.utils.data.DataLoader`|Dataset을 배치 반복자로 감싸는 클래스|`batch_size`, `shuffle`, `num_workers`, `pin_memory`|[[Training/Training Loop Patterns]]|
|`torch.utils.data.Subset`|데이터셋 일부만 선택|인덱스 리스트 전달|[[Evaluation/Validation and Early Stopping]]|
|`torch.utils.data.ConcatDataset`|여러 데이터셋을 순차 결합|동일 형식 Dataset만 가능|[[Projects/Project Template]]|
|`Sampler` 계열|샘플링 순서를 제어|`RandomSampler`, `WeightedRandomSampler`|[[Training/Regularization Techniques]]|

## 실습 예제
```python
import torch
from torch.utils.data import Dataset, DataLoader

class VectorDataset(Dataset):
    def __init__(self, length=1000):
        self.length = length
        self.data = torch.randn(length, 16)
        self.labels = torch.randint(0, 2, (length,))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "x": self.data[idx],
            "y": self.labels[idx]
        }

dataset = VectorDataset()
dataloader = DataLoader(dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)

for batch in dataloader:
    x, y = batch["x"], batch["y"]
    print(x.shape, y.shape)
    break
```

## 실수 주의
- `num_workers`를 0보다 크게 설정하면 Windows나 Jupyter 환경에서 `if __name__ == "__main__":` 가드가 필요합니다.
- `pin_memory=True`는 GPU 학습 시에만 유효하며, CPU 전용 학습에서는 성능 향상이 없습니다.
- Dataset에서 반환하는 텐서는 GPU로 옮기기 전에 `.to(device)`를 호출해야 합니다.
- 가변 길이 데이터는 collate 함수(`collate_fn`)를 정의해 배치 형태를 통일해야 합니다.

## 관련 노트
- [[Tensor 생성]]
- [[Tensor 입출력]]
- [[Training/Training Loop Patterns]]
