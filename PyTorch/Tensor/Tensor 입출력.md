## TL;DR
- 텐서 자체를 저장하거나 모델의 `state_dict`를 저장하는 방법을 정리했습니다.
- `torch.save` / `torch.load`는 Python pickle 기반이라 버전 호환성과 보안에 주의해야 합니다.
- 디바이스가 다른 환경에서 로드할 때는 `map_location`으로 안전하게 매핑하세요.

## 언제 쓰나
- 학습 중 체크포인트를 저장하고 중단 지점에서 재개할 때
- 사전 계산된 텐서를 캐시하거나 공유할 때
- NumPy, CSV 등 외부 포맷과 상호 변환할 때

## 주요 패턴
|패턴|코드|설명|
|---|---|---|
|텐서 저장|`torch.save(tensor, 'foo.pt')`|pickle 직렬화|
|텐서 로드|`tensor = torch.load('foo.pt', map_location='cpu')`|다른 device 매핑 가능|
|모델 state 저장|`torch.save(model.state_dict(), 'model.pt')`|구조 일치 필요|
|체크포인트 저장|`torch.save({'epoch': ..., 'model': model.state_dict()}, 'ckpt.pt')`|학습 상태 묶음|
|체크포인트 로드|`checkpoint = torch.load('ckpt.pt'); model.load_state_dict(checkpoint['model'])`|추가 정보 활용|
|NumPy 변환|`np.save('x.npy', tensor.cpu().numpy())`|CPU 텐서만 가능|

## 실습 예제
```python
import torch

weights = torch.randn(3, 3)
torch.save(weights, 'weights.pt')

loaded = torch.load('weights.pt', map_location='cpu')
print(torch.allclose(weights, loaded))

model = torch.nn.Linear(10, 5)
torch.save({'epoch': 10,
            'model': model.state_dict()},
           'checkpoint.pt')

checkpoint = torch.load('checkpoint.pt', map_location='cpu')
model.load_state_dict(checkpoint['model'])
start_epoch = checkpoint['epoch'] + 1
```

## 실수 주의
- GPU에서 저장한 텐서를 CPU 환경에서 로드할 때는 `map_location='cpu'`를 지정하세요.
- pickle 기반이라 신뢰할 수 없는 소스의 파일은 로드하지 마세요.
- 아주 큰 텐서는 저장 경로에 충분한 디스크 공간이 있는지 확인하고, 필요하면 `pickle_protocol`을 낮춰 구버전과 호환하세요.

## 관련 노트
- [[Tensor 생성]]
- [[Tensor 변환]]
- [[Tensor 가속]]
