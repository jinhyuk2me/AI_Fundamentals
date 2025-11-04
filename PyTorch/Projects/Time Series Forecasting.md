## TL;DR
- 시계열 예측 파이프라인은 시계열 윈도우 생성 → 모델 학습(RNN/Transformer/Temporal CNN) → 예측 및 평가 순으로 진행됩니다.
- 정규화, 누락값 처리, 스케일 변환, 계절성 제거 등 전처리가 성능에 큰 영향을 줍니다.
- MSE/MAE 등 회귀 지표 외에도 MAPE, SMAPE처럼 시계열 특화 지표를 함께 확인하세요.

## 언제 쓰나
- 판매량, 센서 데이터, 교통량 등 시간에 따른 연속 값을 예측해야 할 때
- Multivariate 시계열이나 다중 스텝 예측을 수행하고자 할 때
- 기존 통계 모델(ARIMA 등)보다 딥러닝 접근을 테스트해보고 싶을 때

## 단계별 구성
|단계|설명|주요 작업|관련 노트|
|---|---|---|---|
|데이터 준비|윈도우/시퀀스 생성, 정규화|`sliding_window`, 누락값 처리|[[Data/Dataset and DataLoader]]|
|모델|RNN, LSTM, GRU, Temporal CNN, Transformer|`nn.LSTM`, `nn.Transformer`|[[Modeling/Core Layers]]|
|학습|회귀 손실, 옵티마이저|MSELoss + Adam|[[Training/Loss Functions]], [[Training/Optimizers and Schedulers]]|
|평가|MAE, RMSE, MAPE|슬라이딩 예측, 롤링 검증|[[Evaluation/Metrics]], [[Evaluation/Validation and Early Stopping]]|
|배포|실시간/배치 inference|TorchScript, batch API|[[Deployment/Serving Options]]|

## 데이터셋 예제
```python
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, series, lookback, horizon):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.series) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        window = self.series[idx:idx + self.lookback]
        target = self.series[idx + self.lookback:
                              idx + self.lookback + self.horizon]
        return window.unsqueeze(-1), target
```

## 모델 예제(LSTM)
```python
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)
```

## 실수 주의
- 시계열 데이터는 순서를 유지해야 하므로 학습/검증 분할 시 미래 데이터를 학습에 포함시키지 않도록 주의하세요.
- 입력 시퀀스를 표준화/정규화했다면 예측 결과를 역변환(inverse transform)해 실제 단위로 해석하세요.
- 다중 스텝 예측 시 teacher forcing을 사용할지, autoregressive 방식을 사용할지 결정해야 합니다.
- 계절성과 추세를 모델링하기 위해 추가 특징(요일, 월, 이벤트 등)을 포함시키는 것이 유리합니다.

## 관련 노트
- [[torchaudio and torchtext]]
- [[Training/Regularization Techniques]]
- [[Evaluation/Validation and Early Stopping]]
