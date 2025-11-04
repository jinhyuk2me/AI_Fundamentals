## TL;DR
- 모델 서빙은 배포 환경(REST, gRPC, 배치, 스트리밍)에 맞는 런타임과 인프라 선택이 핵심입니다.
- TorchServe, FastAPI/Flask + PyTorch, Triton Inference Server, ONNX Runtime 등이 대표 옵션입니다.
- 배포 전 모델 최적화(quantization, compile), 버전 관리, 모니터링 전략을 준비해야 안정적으로 운영할 수 있습니다.

## 언제 쓰나
- 학습된 PyTorch 모델을 실서비스에 적용하거나 API로 제공할 때
- 대규모 요청을 처리해야 하거나, GPU 리소스를 효율적으로 활용하고 싶을 때
- 다양한 하드웨어/플랫폼에서 추론을 수행해야 할 때

## 주요 서빙 옵션
|옵션|특징|장점|주의점|
|---|---|---|---|
|TorchServe|PyTorch 공식 서빙 프레임워크|모델 아카이브, 배치 처리 지원|핸들러 구현 필요|
|FastAPI/Flask|파이썬 웹 프레임워크|커스터마이징 자유도 높음|스케일링 직접 구성|
|Triton Inference Server|NVIDIA 서빙 플랫폼|다중 프레임워크 지원, GPU 최적화|ONNX/TensorRT 변환 필요|
|ONNX Runtime|ONNX 모델 추론 엔진|다양한 디바이스 지원|사전 변환 단계 필요|

## TorchServe 흐름
1. 모델을 TorchScript 또는 eager 모드로 저장 (`model.pt`).
2. `handler.py`에서 전처리/후처리 로직 구현.
3. `torch-model-archiver`로 `.mar` 파일 생성.
4. `torchserve --start --model-store ./model_store --models mymodel=mymodel.mar` 실행.

## FastAPI 예제
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.jit.load("model.pt").eval()

@app.post("/predict")
def predict(payload: dict):
    tensor = preprocess(payload)
    with torch.no_grad():
        output = model(tensor)
    return postprocess(output)
```

## Triton 배포 개요
- 모델을 ONNX 또는 TensorRT 형식으로 변환.
- 모델 리포지토리를 구조에 맞게 구성 (`model_repository/model_name/1/model.onnx`).
- `tritonserver --model-repository=./model_repository`로 서버 실행.
- HTTP/gRPC 클라이언트를 통해 추론 요청.

## 실수 주의
- CPU/GPU 디바이스를 명시적으로 고정하거나, 멀티프로세스 환경에서 CUDA 초기화를 주의하세요.
- TorchServe 핸들러에서 전역 변수로 모델을 로드하고, 요청마다 불필요한 초기화가 반복되지 않도록 합니다.
- FastAPI/Flask를 사용할 때는 Uvicorn/Gunicorn과 같은 ASGI/WSGI 서버를 구성하고, 로드 밸런싱을 고려해야 합니다.
- 서빙 모델의 버전을 관리하고, 롤백 전략(예: Blue-Green, Canary)을 마련해 두어야 합니다.

## 관련 노트
- [[Deployment/Mobile and Edge]]
- [[Deployment/Model Versioning]]
- [[Advanced/Graph and Compilation]]
