## TL;DR
- NLP 파이프라인은 텍스트 정제·토큰화 → 임베딩 → 모델 학습(순환/Transformer) → 평가 → 배포 단계로 구성됩니다.
- 토크나이저와 단어 사전을 적절히 구성하고, 패딩/마스킹을 일관되게 적용해야 합니다.
- 전이학습(BERT 등)을 사용할지, 경량 모델을 직접 학습할지에 따라 파이프라인을 조정하세요.

## 언제 쓰나
- 텍스트 분류, 감성 분석, 질의응답 등 자연어 처리 문제를 해결할 때
- 도메인 특화 말뭉치로 임베딩을 학습하거나 파인튜닝이 필요할 때
- 실시간/배치 추론 서비스를 구축하고자 할 때

## 단계별 구성
|단계|설명|주요 작업|관련 노트|
|---|---|---|---|
|데이터 정제|토큰화, 필터링, 사전 구축|`torchtext` 또는 HuggingFace Tokenizer|[[torchaudio and torchtext]]|
|임베딩|사전학습 임베딩, subword|`nn.Embedding`, `AutoTokenizer`|[[Modeling/Core Layers]]|
|모델|RNN, CNN, Transformer|`nn.LSTM`, `nn.TransformerEncoder`, `BERT`|[[Modeling/Functional API]]|
|학습|CrossEntropyLoss, AdamW|패딩 마스크, 학습률 스케줄|[[Training/Loss Functions]], [[Training/Optimizers and Schedulers]]|
|평가|Accuracy, F1, BLEU 등|Confusion Matrix, 라벨별 리포트|[[Evaluation/Metrics]]|
|배포|Tokenizer + 모델 직렬화|FastAPI, TorchServe|[[Deployment/Serving Options]]|

## 토크나이저 & DataLoader 예제
```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset

tokenizer = get_tokenizer("basic_english")

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        tokens = tokenizer(text)
        ids = self.vocab(tokens)
        return torch.tensor(ids, dtype=torch.long), label

def collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), lengths
```

## BERT 파인튜닝 틀
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=num_classes
)

inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs, labels=batch_labels)
loss = outputs.loss
```

## 실수 주의
- Tokenizer와 모델이 사용하는 vocabulary가 일치해야 합니다. 파인튜닝 시 토크나이저를 함께 저장하세요.
- 패딩 토큰이 손실 계산에 포함되지 않도록 `attention_mask` 또는 `ignore_index`를 활용하세요.
- 긴 문장을 처리할 때는 `max_length`와 `truncation` 설정을 명확히 하고, 필요한 경우 슬라이딩 윈도우를 사용하세요.
- 배포 시 토크나이저 초기화 비용이 크므로 애플리케이션 시작 시 로드하고 재사용하세요.

## 관련 노트
- [[torchaudio and torchtext]]
- [[Training/Training Loop Patterns]]
- [[Deployment/Serving Options]]
