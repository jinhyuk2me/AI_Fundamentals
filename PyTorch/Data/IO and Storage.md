## TL;DR
- PyTorch는 텐서·모델을 `torch.save`/`torch.load`로 직렬화하고, 외부 포맷과 연동할 때는 NumPy·pandas 등을 활용합니다.
- 대규모 데이터는 스트리밍, 캐싱, 메모리 매핑 기법으로 처리해 I/O 병목을 최소화해야 합니다.
- 환경에 따라 CPU/GPU 장치 매핑을 명시해 복원 오류를 방지합니다.

## 언제 쓰나
- 학습 중 체크포인트 저장 및 복원 파이프라인을 구성할 때
- 사전 계산한 텐서를 디스크에 캐시하거나 다른 프로젝트와 공유할 때
- CSV, 이미지, HDF5 등 외부 데이터 포맷을 PyTorch Dataset으로 연결할 때

## 주요 패턴
|패턴|코드|설명|주의점|
|---|---|---|---|
|텐서 저장|`torch.save(tensor, path)`|pickle 기반 직렬화|신뢰할 수 없는 파일은 로드 금지|
|텐서 로드|`torch.load(path, map_location='cpu')`|디바이스 매핑 가능|GPU → CPU 이동 필요 시 필수|
|모델 상태 저장|`torch.save(model.state_dict(), path)`|구조 동일해야 복원 가능|모델 클래스를 함께 관리|
|체크포인트 묶음|`torch.save({'epoch':..., 'model':...}, path)`|학습 상태, 옵티마이저 포함|키 이름 일관성 유지|
|외부 포맷|`np.save`, `h5py.File`, `pandas.read_csv`|Dataset에서 활용|CPU 텐서로 변환 후 사용|

## 실습 예제
```python
import torch

weights = torch.randn(4, 4)
torch.save(weights, "weights.pt")

loaded = torch.load("weights.pt", map_location="cpu")
assert torch.allclose(weights, loaded)

model = torch.nn.Linear(16, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

checkpoint = {
    "epoch": 5,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict()
}
torch.save(checkpoint, "checkpoint.pt")

restored = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(restored["model"])
optimizer.load_state_dict(restored["optimizer"])
start_epoch = restored["epoch"] + 1
```

## 실수 주의
- GPU에서 저장한 텐서를 CPU에서 로드할 때는 `map_location="cpu"`를 지정해야 합니다.
- pickle 기반 직렬화는 버전별 동작이 다를 수 있으므로 PyTorch 버전을 함께 기록하세요.
- 외부 파일 형식은 I/O 시간이 길 수 있으니, 필요한 경우 `Dataset` 초기화 시 인메모리 로드 또는 캐시를 고려하세요.
- `state_dict`를 로드할 때 키가 다르면 `strict=False`를 써야 하지만, 예상치 못한 누락을 일으킬 수 있으므로 로그를 확인하세요.

## 관련 노트
- [[Tensor 입출력]]
- [[Training/Training Loop Patterns]]
- [[Deployment/Serving Options]]
