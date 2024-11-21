# transformer_block.py
import torch
from torch import Tensor
from typing import Optional
from mlgroup1.scripts.multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from mlgroup1.scripts.feed_forward import FeedForward
from nn_objects2 import Activation

class TransformerBlock:
    def __init__(self, embed_size: int, heads: int, forward_expansion: int, activation: Activation):
        """
        Transformer 블록 초기화

        Args:
            embed_size (int): 임베딩 차원
            heads (int): 어텐션 헤드 수
            forward_expansion (int): 피드포워드 네트워크 확장 비율
            activation (Activation): 활성화 함수
        """
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion, activation)
        self.norm2 = LayerNorm(embed_size)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [batch_size, seq_length, embed_size]
            mask (Optional[Tensor], optional): 마스크 텐서. Defaults to None.

        Returns:
            Tensor: Transformer 블록 출력
        """
        # Multi-Head Attention
        attention_out = self.attention.forward(x, mask)
        # 잔차 연결 및 레이어 정규화
        x = self.norm1.forward(x + attention_out)
        # 피드포워드 네트워크
        ff_out = self.feed_forward.forward(x)
        # 잔차 연결 및 레이어 정규화
        out = self.norm2.forward(x + ff_out)
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트
        """
        # 잔차 연결 역전파
        grad = self.norm2.backward(grad_output)
        grad = self.feed_forward.backward(grad)
        grad = grad + grad_output  # 잔차 연결

        # 피드포워드 네트워크 역전파
        grad = self.norm1.backward(grad)
        grad = self.attention.backward(grad)  # MultiHeadAttention.backward 호출
        grad = grad + grad_output  # 잔차 연결
        return grad
