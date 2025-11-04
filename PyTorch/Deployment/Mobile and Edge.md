## TL;DR
- 모바일·엣지 배포를 위해 TorchScript, PyTorch Mobile, ONNX Runtime Mobile, TensorRT 등 경량 런타임을 활용합니다.
- 모델 용량 축소와 추론 속도 향상을 위해 quantization, pruning, knowledge distillation을 적용합니다.
- 플랫폼별 빌드 도구(Android NDK, iOS toolchain)와 인터페이스를 준비해야 합니다.

## 언제 쓰나
- 스마트폰, 임베디드 기기, IoT 환경에서 실시간 추론을 수행할 때
- 네트워크 연결이 제한되어 온디바이스 추론이 필요한 경우
- 에너지 소비와 메모리 사용량을 최소화해야 할 때

## 배포 선택지
|옵션|설명|장점|주의점|
|---|---|---|---|
|PyTorch Mobile|TorchScript 기반 모바일 런타임|PyTorch 코드와 일관성|앱 크기 증가 가능|
|ONNX Runtime Mobile|경량 ONNX 엔진|다양한 플랫폼 지원|사전 변환 필요|
|TensorRT|NVIDIA GPU 엣지 디바이스|FP16/INT8 최적화|NVIDIA 하드웨어 의존|
|Core ML|iOS 전용 모델 포맷|애플 생태계 최적화|변환 과정 필요|

## PyTorch Mobile 흐름
1. 모델을 TorchScript(`torch.jit.script`/`trace`)로 변환.
2. `pt-mobile` 빌드 설정을 적용해 Android/iOS 프로젝트에 포함.
3. Java/Kotlin/Swift에서 `LiteModuleLoader`로 모델 로드.
4. 입력 전처리·후처리 로직을 네이티브 코드로 구현.

### 간단한 변환 예
```python
import torch
from torchvision import models

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

scripted = torch.jit.script(model)
scripted.save("mobilenet_v2.pt")
```

## 최적화 기법
|기법|효과|적용 단계|
|---|---|---|
|Quantization|INT8 등 정수 연산으로 속도 개선|학습 후(post-training) 또는 QAT|
|Pruning|가중치 제거로 모델 크기 축소|학습 중 또는 후처리|
|Knowledge Distillation|경량 모델에 대형 모델 지식 전달|학습 단계|

## 실수 주의
- TorchScript로 변환되지 않는 Python 기능(동적 속성, 일부 라이브러리)을 사용하면 스크립팅이 실패할 수 있습니다.
- 모바일 빌드 시 ABI, CPU 아키텍처를 명확히 지정해야 런타임 오류를 피할 수 있습니다.
- 엣지 디바이스는 메모리 제한이 크므로, 배치 크기 1 추론을 기본으로 테스트하세요.
- 배포 후 실제 디바이스에서 온디바이스 테스트를 수행해 성능과 전력 소비를 검증해야 합니다.

## 관련 노트
- [[Deployment/Serving Options]]
- [[Advanced/AMP and Quantization]]
- [[Projects/Project Template]]
