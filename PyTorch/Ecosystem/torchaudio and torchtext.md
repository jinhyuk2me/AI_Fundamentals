## TL;DR
- `torchaudio`와 `torchtext`는 각각 오디오·텍스트 데이터를 처리하는 PyTorch 공식 라이브러리입니다.
- 데이터셋, 변환, 사전학습 모델을 제공해 음성 인식·텍스트 분류 등 다양한 작업을 빠르게 시작할 수 있습니다.
- 토크나이저, 스펙트로그램, 데이터 파이프라인 구성을 이해하면 커스텀 모델과 쉽게 연동됩니다.

## torchaudio 핵심 요소
|구성|설명|예시|관련 노트|
|---|---|---|---|
|데이터셋|`torchaudio.datasets.LIBRISPEECH` 등|음성 인식용 공개 데이터|[[Data/Dataset and DataLoader]]|
|변환|`torchaudio.transforms.MelSpectrogram`|오디오 → 스펙트로그램|[[Data/Transforms/Transforms Overview]]|
|모델|`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`|사전학습 ASR 모델|[[Projects/Time Series Forecasting]]|
|I/O|`torchaudio.load`, `torchaudio.save`|Waveform 입출력|[[Data/IO and Storage]]|

### torchaudio 예제
```python
import torchaudio

waveform, sample_rate = torchaudio.load("speech.wav")
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=80
)(waveform)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().eval()
with torch.no_grad():
    emission, _ = model(waveform)
```

## torchtext 핵심 요소
|구성|설명|예시|관련 노트|
|---|---|---|---|
|데이터셋|`torchtext.datasets.AG_NEWS`, `IMDB`|텍스트 분류/요약|[[Data/Dataset and DataLoader]]|
|텍스트 변환|Tokenizer, Vocab|`torchtext.data.utils.get_tokenizer`|[[Projects/NLP Pipeline]]|
|사전학습 임베딩|`GloVe`, `FastText`|`torchtext.vocab.GloVe`|[[Modeling/Core Layers]]|
|데이터 파이프|`torchtext.data.functional`|스트림 방식 전처리|[[Data/Transforms/Transforms Overview]]|

### torchtext 예제
```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = torchtext.datasets.AG_NEWS(split="train")
vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                  specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
```

## 실수 주의
- torchaudio 변환은 입력 채널 수와 샘플링 주파수를 맞춰야 하며, stereo → mono 변환이 필요한 경우가 많습니다.
- torchtext의 Tokenizer/Vocab을 DataLoader와 함께 사용할 때는 `collate_fn`에서 패딩을 처리해야 합니다.
- 사전학습 음성 모델은 GPU 메모리를 많이 사용하므로 batch size를 작은 값부터 조정하세요.
- 텍스트 데이터는 인코딩 문제(UTF-8 등)를 사전에 확인하고, 정제(cleaning) 단계가 필요할 수 있습니다.

## 관련 노트
- [[Projects/Time Series Forecasting]]
- [[Projects/NLP Pipeline]]
- [[Training/Training Loop Patterns]]
