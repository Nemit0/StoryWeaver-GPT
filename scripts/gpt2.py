# gpt2.py
import torch
from torch import Tensor
from typing import Optional
from mlgroup1.model.Embedding import Embedding
from mlgroup1.model.positional_encoding import PositionalEncoding
from mlgroup1.scripts.transformer_block import TransformerBlock
from nn_objects2 import Activation, Layer, ReLU, Linear
from layer_norm import LayerNorm

class GPT2:
    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 512,
                 max_seq_len: int = 512,
                 num_layers: int = 6,
                 heads: int = 8,
                 forward_expansion: int = 4,
                 dropout: float = 0.1):
        """
        GPT-2 모델 초기화

        Args:
            vocab_size (int): 어휘 집합 크기
            embed_size (int, optional): 임베딩 차원. Defaults to 512.
            max_seq_len (int, optional): 최대 시퀀스 길이. Defaults to 512.
            num_layers (int, optional): Transformer 블록 수. Defaults to 6.
            heads (int, optional): 어텐션 헤드 수. Defaults to 8.
            forward_expansion (int, optional): 피드포워드 네트워크 확장 비율. Defaults to 4.
            dropout (float, optional): 드롭아웃 비율. Defaults to 0.1.
        """
        self.embed_size = embed_size
        self.embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(max_seq_len, embed_size)
        self.transformer_blocks = [TransformerBlock(embed_size, heads, forward_expansion, ReLU()) for _ in range(num_layers)]
        self.layer_norm = LayerNorm(embed_size)
        self.fc_out = Layer(embed_size, vocab_size, Linear())

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 토큰 인덱스 [batch_size, seq_length]
            mask (Optional[Tensor], optional): 마스크 텐서. Defaults to None.

        Returns:
            Tensor: 출력 logits [batch_size, seq_length, vocab_size]
        """
        # 임베딩
        x = self.embedding.forward(x)  # [batch_size, seq_length, embed_size]
        # 위치 인코딩
        x = self.positional_encoding.forward(x)  # [batch_size, seq_length, embed_size]

        # Transformer 블록 통과
        for block in self.transformer_blocks:
            x = block.forward(x, mask)  # [batch_size, seq_length, embed_size]

        # 최종 레이어 정규화
        x = self.layer_norm.forward(x)  # [batch_size, seq_length, embed_size]

        # 출력 레이어
        logits = self.fc_out.forward(x)  # [batch_size, seq_length, vocab_size]

        return logits

    def backward(self, grad_output: Tensor) -> None:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트
        """
        # 출력 레이어 역전파
        grad = self.fc_out.backward(grad_output)  # [batch_size, seq_length, embed_size]
        # 최종 레이어 정규화 역전파
        grad = self.layer_norm.backward(grad)
        # Transformer 블록 역전파
        for block in reversed(self.transformer_blocks):
            grad = block.backward(grad)  # [batch_size, seq_length, embed_size]
        # 위치 인코딩 역전파 (생략)
        # 임베딩 역전파
        self.embedding.backward(grad)
