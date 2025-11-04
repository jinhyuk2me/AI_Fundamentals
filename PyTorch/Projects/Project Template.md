## TL;DR
- 샘플 프로젝트는 데이터 준비 → 모델 구성 → 학습 → 평가 → 배포까지 일관된 흐름으로 정리합니다.
- 각 단계에서 사용한 노트와 코드 스니펫을 링크하면 복습과 재사용이 쉬워집니다.
- 프로젝트 폴더에는 README, 환경 설정, 스크립트, 로그/체크포인트 경로를 명시합니다.

## 언제 쓰나
- 새로운 도메인 문제를 해결할 때 기본 뼈대를 재활용하고 싶을 때
- 팀원과 프로젝트 구조를 공유하거나 교육 자료로 활용할 때
- 실험을 체계적으로 기록하고 재현성을 높이고자 할 때

## 프로젝트 구조 예시
```
project-name/
├─ README.md
├─ configs/
│  ├─ train.yaml
│  └─ model.yaml
├─ data/
│  └─ raw/ processed/
├─ notebooks/
├─ src/
│  ├─ datasets.py
│  ├─ models.py
│  ├─ train.py
│  └─ evaluate.py
├─ scripts/
│  └─ run_train.sh
├─ logs/
├─ checkpoints/
└─ requirements.txt or pyproject.toml
```

## 진행 체크리스트
|단계|설명|관련 노트|
|---|---|---|
|데이터 준비|수집, 정제, split, 증강|[[Data/Dataset and DataLoader]], [[Data/Transforms/Transforms Overview]]|
|모델 설계|베이스라인 → 개선안|[[Modeling/Module Basics]], [[Modeling/Core Layers]]|
|학습 설정|Optimizer, Scheduler, Loop|[[Training/Training Loop Patterns]], [[Training/Optimizers and Schedulers]]|
|평가 및 분석|메트릭, 시각화, Error analysis|[[Evaluation/Metrics]], [[Evaluation/Logging and Visualization]]|
|배포 계획|서빙, 버전 관리, 모니터링|[[Deployment/Serving Options]], [[Deployment/Model Versioning]]|

## README 초안
```
# 프로젝트 이름

## 개요
- 문제 정의
- 데이터 출처
- 주요 모델/기법

## 환경 세팅
```bash
poetry install
poetry run python -m pip install -r requirements.txt
```

## 학습 실행
```bash
poetry run python src/train.py --config configs/train.yaml
```

## 평가
```bash
poetry run python src/evaluate.py --checkpoint checkpoints/best.pt
```

## 결과
- 주요 메트릭
- 그래프/시각화 링크
- 모델 아티팩트 위치
```

## 실수 주의
- 데이터와 체크포인트 경로를 하드코딩하지 말고 설정 파일이나 환경 변수로 관리하세요.
- 실험이 늘어나면 로그/체크포인트가 쌓이므로 주기적으로 정리하거나 naming을 체계화하세요.
- README/노트에 업데이트 날짜를 명시해, 최신 상태인지 추적하세요.

## 관련 노트
- [[Project Tooling]]
- [[Evaluation/Logging and Visualization]]
- [[Deployment/Model Versioning]]
