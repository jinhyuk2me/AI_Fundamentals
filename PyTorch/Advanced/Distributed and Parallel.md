## TL;DR
- PyTorch는 단일 GPU(DataParallel)부터 다중 노드(DistributedDataParallel, FSDP)까지 다양한 병렬 전략을 제공합니다.
- 분산 환경에서는 초기화, 프로세스 그룹, Sampler 설정을 올바르게 구성해야 합니다.
- DDP를 기본으로 사용하고, 모델 크기에 따라 ZeRO/FSDP/DeepSpeed 등을 검토합니다.

## 언제 쓰나
- 여러 GPU를 활용해 학습 시간을 단축하고자 할 때
- 메모리 한계를 넘어서는 대규모 모델을 학습해야 할 때
- 멀티노드 클러스터에서 동일 코드를 실행하고자 할 때

## 주요 방식 비교
|방식|설명|장점|주의점|
|---|---|---|---|
|`nn.DataParallel`|단일 프로세스에서 입력을 GPU로 나눔|사용법 간단|성능/스케일링 제한, 권장X|
|`DistributedDataParallel (DDP)`|프로세스당 GPU 1개, 파라미터 동기화|PyTorch 권장 방식|Sampler, 초기화 설정 필요|
|`FullyShardedDataParallel (FSDP)`|파라미터를 샤딩해 메모리 분산|초거대 모델 지원|설정 복잡, PyTorch 2.0+|
|ZeRO/DeepSpeed|최적화 상태까지 샤딩|대규모 모델 효율|외부 라이브러리 의존|

## DDP 기본 예제
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)

    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = MyDataset()
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad()
            outputs = ddp_model(batch["x"].to(rank))
            loss = criterion(outputs, batch["y"].to(rank))
            loss.backward()
            optimizer.step()

    cleanup()
```

## 실수 주의
- `DistributedSampler`를 사용하지 않으면 각 프로세스가 동일 데이터를 중복 학습하게 됩니다.
- Environment 변수(`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`)가 제대로 설정되지 않으면 프로세스 그룹 초기화가 실패합니다.
- DataParallel은 GPU 간 통신이 비효율적이어서 DDP로 빠르게 전환하는 것이 좋습니다.
- NCCL backend는 GPU/NVLink 환경에 최적화되어 있으므로 CPU만 사용하는 경우 `gloo`를 사용하세요.

## 관련 노트
- [[Advanced/AMP and Quantization]]
- [[Advanced/Profiling and Debugging]]
- [[Deployment/Serving Options]]
