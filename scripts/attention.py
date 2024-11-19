# attention.py
import torch
from torch import Tensor
from typing import Optional

class ScaledDotProductAttention:
    def __init__(self, embed_size: int):
        """
        Scaled Dot-Product Attention 초기화

        Args:
            embed_size (int): 임베딩 차원
        """
        self.embed_size = embed_size
        self.scale = embed_size ** 0.5  # 스케일링 인자

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        순전파 과정

        Args:
            Q (Tensor): Query 행렬 [batch_size, heads, seq_length, head_dim]
            K (Tensor): Key 행렬 [batch_size, heads, seq_length, head_dim]
            V (Tensor): Value 행렬 [batch_size, heads, seq_length, head_dim]
            mask (Optional[Tensor], optional): 마스크 텐서. Defaults to None.

        Returns:
            Tensor: Attention 출력 [batch_size, heads, seq_length, head_dim]
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

        # Q와 K의 내적을 통해 Attention 점수 계산
        self.scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, heads, seq_length, seq_length]

        if mask is not None:
            self.scores = self.scores.masked_fill(mask == 0, float('-inf'))

        # 소프트맥스를 통해 Attention 가중치 계산
        self.attention_weights = torch.softmax(self.scores, dim=-1)  # [batch_size, heads, seq_length, seq_length]

        # Attention 가중치를 V에 적용하여 최종 출력 생성
        self.out = torch.matmul(self.attention_weights, V)  # [batch_size, heads, seq_length, head_dim]

        return self.out

    def backward(self, grad_output: Tensor) -> tuple:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트 [batch_size, heads, seq_length, head_dim]

        Returns:
            tuple: (grad_Q, grad_K, grad_V)
        """
        # Gradients for V
        grad_V = torch.matmul(self.attention_weights.transpose(-2, -1), grad_output)  # [batch_size, heads, seq_length, head_dim]

        # Gradients for attention_weights
        grad_attention_weights = torch.matmul(grad_output, self.V.transpose(-2, -1))  # [batch_size, heads, seq_length, seq_length]

        # Gradient of softmax (scores)
        grad_scores = self.attention_weights * (grad_attention_weights - (self.attention_weights * grad_attention_weights).sum(dim=-1, keepdim=True))

        # Gradients for Q and K
        grad_Q = torch.matmul(grad_scores, self.K) / self.scale  # [batch_size, heads, seq_length, head_dim]
        grad_K = torch.matmul(grad_scores.transpose(-2, -1), self.Q) / self.scale  # [batch_size, heads, seq_length, head_dim]

        return grad_Q, grad_K, grad_V