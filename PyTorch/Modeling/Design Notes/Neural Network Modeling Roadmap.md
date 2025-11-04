## TL;DR
- 신경망 모델링은 **문제 정의 → 데이터 파악 → 모델 가설 수립 → 검증/고도화**의 반복적인 사이클입니다.
- 구조 선택은 목표(정확도, 속도, 메모리), 데이터 특성(크기, 노이즈, 도메인)과 제약(리소스, 배포 환경)에 기반해야 합니다.
- 실험 로그와 가설을 문서화하면 추후 개선과 회고가 쉬워집니다.

## 1. 문제 정의 & 제약 파악
- **목표 지표**: 정확도, F1, mAP, Latency 등 최적화 대상 명확히 하기
- **리소스 제약**: GPU 메모리, 추론 시간, 배포 환경(모바일/서버)
- **데이터 제약**: 샘플 수, 클래스 불균형, 라벨 품질, 입력 포맷
- **명시적 가설**: “Residual block으로 깊이를 늘리면 성능 향상”, “작은 모델이라도 데이터 증강으로 성능 확보” 등

## 2. 데이터 이해 & 준비
- **탐색(EDA)**: 입력 분포, 아웃라이어, 시각적 패턴 확인
- **전처리**: 정규화, 누락값 처리, 토큰화/스펙트로그램 등 도메인별 변환
- **증강 전략**: 학습 데이터 확장, 인위적 노이즈, Mixup/CutMix 등
- **Dataset 설계**: 학습/검증/테스트 분할 기준, 샘플링 전략

## 3. 모델 설계 프로세스
|단계|핵심 질문|실행 아이디어|관련 노트|
|---|---|---|---|
|Baseline 구축|가장 단순한 모델로 목적을 달성할 수 있는가?|Linear/MLP, 작은 CNN/RNN|[[Modeling/Module Basics]]|
|구조 선택|데이터 특성에 맞는 모듈은 무엇인가?|CNN, RNN, Transformer, Hybrid|[[Modeling/Core Layers]]|
|블록 설계|어떤 패턴을 활용할 것인가?|Residual, Bottleneck, Attention|[[Modeling/Custom Blocks]], [[Modeling/Design Notes/Residual Block]], [[Modeling/Design Notes/Bottleneck Residual Block]], [[Modeling/Design Notes/Inception Block]], [[Modeling/Design Notes/Squeeze-Excitation Block]], [[Modeling/Design Notes/Transformer Encoder Block]]|
|정규화 & 안정화|학습 안정성 확보?|BatchNorm, Dropout, Skip, Residual scaling|[[Training/Regularization Techniques]]|
|용량 조절|파라미터 수, 깊이/폭|depth/width scaling, bottleneck|[[Advanced/Distributed and Parallel]]|

### 설계 체크리스트
- 입력/출력 Shape, 채널 수, stride/padding 계산
- 활성 함수 및 정규화 배치 순서 결정
- 초기화 전략: He/Xavier, LayerNorm 기반 등
- 파라미터 수와 FLOPs 추산 (예: `torchinfo.summary`, fvcore)
- 배포 대상(모바일, 서버)에 맞는 연산 사용 여부 확인

## 4. 학습 전략 & 검증
- **Optimizer/스케줄**: AdamW, SGD + warmup, OneCycle 등
- **Loss 선택**: CrossEntropy, Focal, Label Smoothing 등
- **메트릭**: 목표 지표 중심으로 logging, TensorBoard/W&B 활용
- **Early stopping**: 과적합 지속 여부 모니터링, patience 설정
- **실험 로그**: 하이퍼파라미터, seed, 환경 정보, 결과 기록

## 5. 개선 사이클
1. **문제 식별**: 과적합/과소적합, 속도 병목 등
2. **가설 수립**: “Dropout 추가”, “채널 2배 확대”, “Augmentation 강화”
3. **변경 적용**: 코드/설정 변경 후 재학습
4. **평가 & 기록**: 전/후 비교, 실패 시 원인 기록
5. **Refactor**: 재사용 가능한 블록/모듈로 정리

## 6. 배포 고려
- TorchScript/ONNX 변환 가능 여부 테스트
- 지연時間(Latency) 측정, 배치 사이즈 조정
- 모델 버전 관리: `model_vYYYYMMDD` 형식, 메타데이터 기록
- 모니터링 지표 정의: 실시간 정확도, 오류율, 처리량

## 자가 질문
- 이 구조가 해결하려는 문제는 무엇인가?
- 대체 블록/모델 대비 장단점은?
- 데이터량이 10배 증가/감소하면 모델을 어떻게 조정할 것인가?
- 배포 환경에서 추론 시간이 허용 범위 내인가?

## 관련 노트
- [[PyTorch Overview]]
- [[Modeling/Module Basics]]
- [[Training/Training Loop Patterns]]
- [[Evaluation/Logging and Visualization]]
- Papers With Code: https://paperswithcode.com/
