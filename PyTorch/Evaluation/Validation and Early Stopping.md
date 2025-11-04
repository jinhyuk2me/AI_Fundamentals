## TL;DR
- 검증 데이터는 모델이 보지 않은 분포에서 성능을 측정해 과적합 여부를 판단하게 해 줍니다.
- Early stopping은 검증 성능이 개선되지 않을 때 학습을 중단해 시간과 자원을 절약합니다.
- 데이터 분할, 시드 고정, 메트릭 일관성을 유지해야 공정한 비교가 가능합니다.

## 언제 쓰나
- 새로운 모델이나 하이퍼파라미터를 실험하면서 성능을 모니터링할 때
- 데이터가 제한적이라 훈련/검증/테스트 분할이 중요할 때
- 모델 저장 주기와 복원 로직을 구축해야 할 때

## 검증 전략
|전략|설명|장점|주의점|
|---|---|---|---|
|Hold-out|훈련/검증/테스트로 고정 분할|구현 간단|데이터가 적으면 분산 큼|
|K-Fold Cross Validation|K개의 폴드로 나눠 순환 평가|데이터 활용 극대화|연산 비용 증가|
|Stratified Split|클래스 비율 유지 분할|불균형 클래스에 유리|분류 문제에 한정|
|Time Series Split|시간 순서를 고려한 분할|시계열 예측에 적합|과거 정보만 사용|

## Early Stopping 구현 예
```python
class EarlyStopping:
    def __init__(self, patience=5, mode="min", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = None
        self.wait = 0
        self.should_stop = False

    def step(self, metric):
        score = -metric if self.mode == "min" else metric
        if self.best is None or score > self.best + self.delta:
            self.best = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return self.should_stop
```

## 실수 주의
- 검증 데이터는 절대 학습에 사용하지 말고, 하이퍼파라미터 튜닝 후 최종 성능은 테스트 세트에서 확인하세요.
- Early stopping 시점에 저장한 모델을 복원하려면 체크포인트 파일을 별도로 관리해야 합니다.
- 동적 학습률 스케줄러와 early stopping을 함께 사용할 때는 스케줄러가 충분히 학습률을 조정할 시간을 주는 것이 좋습니다.
- K-Fold를 사용할 때는 fold 간 데이터 누락이나 중복이 없는지 확인하고, 매 fold마다 모델을 새로 초기화해야 합니다.

## 관련 노트
- [[Evaluation/Metrics]]
- [[Training/Training Loop Patterns]]
- [[Training/Regularization Techniques]]
