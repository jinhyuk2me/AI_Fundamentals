## TL;DR
- Transformer Encoder Block은 Multi-Head Self-Attention과 Position-wise FFN을 Residual+LayerNorm으로 묶은 구조입니다.
- 시퀀스 전반의 장기 의존성을 O(T^2) 연산으로 한 번에 학습하며, 병렬 처리가 쉬워 딥러닝 표준이 되었습니다.
- BERT, ViT, GPT 등 대부분의 현대 모델에서 기본 빌딩 블록으로 재사용됩니다.

## 언제 쓰나
- 자연어, 시계열, 이미지 패치 등 순서가 있는 입력에서 장거리 의존성을 학습할 때
- RNN 기반 모델에서 병렬 처리 제한으로 학습 속도가 느릴 때
- 기존 CNN/RNN에 Self-Attention을 삽입해 long-range context를 보완하려는 경우

## 핵심 아이디어
- **Multi-Head Self-Attention**: Query/Key/Value를 head 수만큼 분할해 서로 다른 표현 하위 공간을 탐색.
- **Feed-Forward Network**: 각 토큰에 독립적으로 적용되는 2층 MLP로 표현 확장 후 축소.
- **Residual + LayerNorm**: 각 서브 레이어에 skip connection과 정규화를 적용해 안정성 확보.
- **Dropout & Position Encoding**: 과적합 방지, 순서 정보 보존.

## 구조 스케치
```
Input (B, T, C)
 ├─ Multi-Head Self-Attention → Dropout
 ├─ Residual Add + LayerNorm
 ├─ Feed-Forward (Linear → Activation → Linear) → Dropout
 └─ Residual Add + LayerNorm
Output (B, T, C)
```

## 실습 예제
```python
import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_multiplier=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)

        ff_hidden = int(embed_dim * ff_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        attn_out, _ = self.attn(x, x, x,
                                key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask)
        x = self.attn_norm(x + self.attn_dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_out))
        return x
```
- `TransformerEncoderBlock(768, 12)`는 BERT-base의 단일 레이어 구성과 유사하며, 마스크 인자를 전달하면 패딩/미래 토큰을 제한할 수 있습니다.

## 실수 주의
- `num_heads` × `head_dim`이 embed_dim과 일치해야 하므로 설정 시 나누어 떨어지는지 확인하세요.
- 시퀀스 길이가 매우 길 경우 메모리/연산량이 급증하니 Longformer류의 sparse attention을 검토해야 합니다.
- LayerNorm은 입력 텐서를 float32로 유지해야 안정적으로 동작하므로 mixed precision에서 주의가 필요합니다.

## 관련 노트
- Vaswani et al., 2017, "Attention Is All You Need"
- Dosovitskiy et al., 2020, "An Image is Worth 16x16 Words"
- [[Projects/NLP Pipeline]]
- [[Modeling/Core Layers]]
