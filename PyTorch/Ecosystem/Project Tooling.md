## TL;DR
- 프로젝트 규모가 커질수록 환경 관리, 패키지 의존성, 실험 기록, 테스트 자동화 도구가 중요해집니다.
- Poetry/conda, Makefile, pre-commit, pytest 등을 조합하면 재현성과 협업 효율을 확보할 수 있습니다.
- 설정과 스크립트를 문서화해 팀원 간 일관된 워크플로를 유지하세요.

## 언제 쓰나
- 여러 실험을 병행하고 환경을 자주 재구성해야 할 때
- 협업 프로젝트에서 코드 스타일과 품질을 유지하고자 할 때
- 배포 전 파이프라인을 자동화하고 테스트를 체계화하고 싶을 때

## 핵심 도구
|범주|도구|용도|비고|
|---|---|---|---|
|환경 관리|Poetry, conda, pipenv|의존성, 가상환경 관리|`poetry.lock`, `environment.yml`|
|명령 실행|Makefile, `tox`, `nox`|반복 작업 자동화|`make train`, `make lint`|
|코드 품질|pre-commit, black, isort, flake8|포맷팅과 정적 분석|CI와 연동|
|테스트|pytest, hypothesis|단위/통합 테스트|모델 모듈 테스트|
|실험 관리|Weights & Biases, MLflow, TensorBoard|실험 기록|`wandb sweep` 등|

## Makefile 예시
```makefile
.PHONY: install train lint format test

install:
	poetry install

train:
	poetry run python scripts/train.py

lint:
	poetry run flake8 src

format:
	poetry run black src

test:
	poetry run pytest
```

## pre-commit 설정 예시
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

## 실수 주의
- 가상환경을 프로젝트 루트에 고정하고 `.gitignore`에 포함하지 않으면 충돌이 발생할 수 있습니다.
- pre-commit 훅은 설치(`pre-commit install`)를 안 하면 실행되지 않으니, README나 AGENTS에 안내를 남기세요.
- 실험 로그 파일은 용량이 커질 수 있으므로 로그 디렉터리 정리 정책을 마련하세요.
- CI 파이프라인에서 GPU가 없는 환경이라면 CPU/Mock 테스트를 준비하고, 필요하면 데이터 샘플링으로 테스트 시간을 줄이세요.

## 관련 노트
- [[Evaluation/Logging and Visualization]]
- [[Projects/Project Template]]
- [[Lightning and Utilities]]
