## TL;DR
- PyTorch는 동적 계산 그래프와 Pythonic 인터페이스로 연구·실험 친화적인 워크플로를 제공합니다.
- TensorFlow/Keras는 배포·엔터프라이즈 생태계, JAX는 함수형·컴파일 최적화, MXNet은 멀티언어 지원 등 각 프레임워크마다 강점이 다릅니다.
- 프로젝트 목적(연구, 프로덕션, TPU 활용 등)에 따라 프레임워크를 선택하거나 혼합 사용하는 전략이 필요합니다.

## 비교 대상
|프레임워크|핵심 특징|장점|주의점|
|---|---|---|---|
|PyTorch|동적 그래프, Python 우선|직관적 디버깅, 풍부한 라이브러리|장기 지원은 사용자가 관리|
|TensorFlow/Keras|정적 그래프 + 고수준 API|엔터프라이즈 배포, TPU, TFX|그래프 디버깅 난이도|
|JAX|함수형, XLA 컴파일|Autograd + jit로 고성능|PyTorch 대비 생태계 작음|
|MXNet/DeepLearning4J 등|다중 언어, 경량|특정 산업군에 최적|커뮤니티 규모 제한|

## PyTorch 선택 기준
- **연구/프로토타입**: 동적 그래프 덕분에 실험이 빠르고, 디버깅이 용이합니다.
- **생태계**: HuggingFace Transformers, Lightning, Detectron2 등 PyTorch 기반 라이브러리가 풍부합니다.
- **학습 곡선**: Python 중심 설계로 딥러닝 초보자도 접근성이 높습니다.
- **배포**: TorchScript, TorchServe, ONNX 등 선택지가 늘어났지만, 기업용 통합 파이프라인(TFX)에 비해 수동 설정이 필요할 수 있습니다.

## TensorFlow/Keras 비교 메모
- `model.fit()` 기반 고수준 API가 강점이며, TF Lite/TF Serving/TFX로 이어지는 파이프라인이 잘 갖춰져 있습니다.
- 정적 그래프 기반이지만 `tf.function`으로 동적 스타일을 흉내낼 수 있습니다.
- TPU 지원이 공식적이므로 대규모 학습에 유리합니다.

## JAX 비교 메모
- 함수형 프로그래밍 스타일과 `jax.grad`, `jax.jit`, `jax.vmap` 등 강력한 컴포저빌리티를 제공합니다.
- XLA 컴파일로 CPU/GPU/TPU에서 고성능을 구현할 수 있으나, PyTorch보다 레퍼런스 코드와 튜토리얼이 적습니다.
- Flax, Haiku 같은 고수준 라이브러리를 사용하면 구조화를 도울 수 있습니다.

## 프레임워크 혼합 전략
- **PyTorch + ONNX/TensorRT**: 학습은 PyTorch, 추론은 최적화된 엔진에서 수행.
- **PyTorch + TensorFlow Serving**: ONNX/TF 변환을 통해 기존 인프라 활용.
- **PyTorch + JAX**: 연구 단계에서 PyTorch로 빠르게 실험 후, JAX로 리라이팅해 TPU/컴파일 최적화 시도.

## 실수 주의
- 프레임워크 전환 시 모델 가중치, 입력 전처리, 사용자 정의 연산 호환성을 반드시 검증하세요.
- 동일 코드를 두 프레임워크에서 유지하려면 추상화 계층이 필요하므로, 초기 설계 단계에서 선택을 확정하는 것이 효율적입니다.
- 팀 역량과 운영 환경(모니터링, 배포 파이프라인)에 따라 학습 비용이 크게 달라질 수 있습니다.

## 관련 노트
- [[PyTorch Overview]]
- [[Deployment/Serving Options]]
- [[Advanced/Graph and Compilation]]
