## TL;DR
- GPU, CPU, Metal 등 다양한 디바이스를 감지하고 텐서·모델을 옮기는 방법을 정리했습니다.
- 혼합 정밀도(AMP), 핀 메모리, 스트림 등 자주 쓰는 가속 기법의 기본 패턴을 제공합니다.
- 모델과 입력 텐서가 항상 같은 디바이스에 있어야 한다는 점을 기억하세요.

## 언제 쓰나
- GPU 학습/추론 환경을 구성할 때
- DataLoader를 최적화해 CPU→GPU 전송 병목을 줄이고 싶을 때
- AMP로 속도는 높이고 메모리는 절약하고 싶을 때

## 주요 기능
|구분|코드|설명|
|---|---|---|
|디바이스 확인|`torch.cuda.is_available()`|CUDA 가능 여부|
|디바이스 선택|`device = torch.device('cuda', index=0)`|GPU 인덱스 지정|
|모델 이동|`model.to(device)`|파라미터·버퍼 이동|
|텐서 이동|`tensor = tensor.to(device)`|데이터 이동|
|혼합 정밀도|`with torch.cuda.amp.autocast():`|자동 FP16/FP32 혼용|
|Grad 스케일링|`scaler = torch.cuda.amp.GradScaler()`|언더플로 방지|
|핀 메모리|`DataLoader(..., pin_memory=True)`|CPU→GPU 전송 최적화|
|다중 GPU(간단)|`torch.nn.DataParallel(model)`|간단 병렬, 권장X|
|다중 GPU(권장)|`torch.nn.parallel.DistributedDataParallel(model)`|DDP 기반 분산|

## 실습 예제
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.nn.Linear(128, 10).to(device)

optimizer = torch.optim.Adam(model.parameters())
scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

for batch in dataloader:
    inputs = batch['x'].to(device, non_blocking=True)
    targets = batch['y'].to(device, non_blocking=True)

    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 실수 주의
- 모델과 입력 텐서가 다른 디바이스에 있으면 `RuntimeError`가 발생합니다. 로드 직후 `model.to(device)`를 습관화하세요.
- AMP 사용 시 나눗셈/로그 등 민감한 연산에서 언더플로가 생길 수 있습니다. 필요하면 `.float()`로 캐스팅하거나 GradScaler를 활용하세요.
- DataParallel보다 DistributedDataParallel이 효율적입니다. 단일 GPU라면 감싸지 않아도 됩니다.

## 관련 노트
- [[Tensor 변환]]
- [[Tensor 입출력]]
- [[Tensor Autograd]]
