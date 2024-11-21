# multi_head_attention.py
import torch
from torch import Tensor
from typing import Optional
from mlgroup1.scripts.attention import ScaledDotProductAttention

class MultiHeadAttention:
    def __init__(self, embed_size: int, heads: int):
        """
        Multi-Head Attention 초기화

        Args:
            embed_size (int): 임베딩 차원
            heads (int): 어텐션 헤드 수
        """
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "임베딩 차원(embed_size)은 헤드 수(heads)로 나누어 떨어져야 합니다."

        # Q, K, V 선형 변환을 위한 가중치 초기화
        self.W_Q = torch.randn(embed_size, embed_size) * (self.head_dim ** -0.5)
        self.W_K = torch.randn(embed_size, embed_size) * (self.head_dim ** -0.5)
        self.W_V = torch.randn(embed_size, embed_size) * (self.head_dim ** -0.5)
        self.W_O = torch.randn(embed_size, embed_size) * (self.head_dim ** -0.5)

        # Attention 메커니즘 초기화
        self.attention = ScaledDotProductAttention(self.head_dim)

        # Initialize gradients
        self.grad_W_Q = torch.zeros_like(self.W_Q)
        self.grad_W_K = torch.zeros_like(self.W_K)
        self.grad_W_V = torch.zeros_like(self.W_V)
        self.grad_W_O = torch.zeros_like(self.W_O)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [batch_size, seq_length, embed_size]
            mask (Optional[Tensor], optional): 마스크 텐서. Defaults to None.

        Returns:
            Tensor: Multi-Head Attention 출력 [batch_size, seq_length, embed_size]
        """
        self.x = x  # [batch_size, seq_length, embed_size]
        self.mask = mask

        # Q, K, V 생성
        self.Q = torch.matmul(x, self.W_Q)  # [batch_size, seq_length, embed_size]
        self.K = torch.matmul(x, self.W_K)  # [batch_size, seq_length, embed_size]
        self.V = torch.matmul(x, self.W_V)  # [batch_size, seq_length, embed_size]

        # 헤드 수에 맞게 분할
        batch_size, seq_length, embed_size = x.size()
        self.Q = self.Q.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)  # [batch_size, heads, seq_length, head_dim]
        self.K = self.K.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)  # [batch_size, heads, seq_length, head_dim]
        self.V = self.V.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)  # [batch_size, heads, seq_length, head_dim]

        # Attention 계산
        self.attention_out = self.attention.forward(self.Q, self.K, self.V, mask)  # [batch_size, heads, seq_length, head_dim]

        # 헤드 결합
        self.attention_out = self.attention_out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)  # [batch_size, seq_length, embed_size]

        # 최종 선형 변환
        out = torch.matmul(self.attention_out, self.W_O)  # [batch_size, seq_length, embed_size]
        print(f"MultiHeadAttention forward input shape: {x.shape}")
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트 [batch_size, seq_length, embed_size]

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트 [batch_size, seq_length, embed_size]
        """
        batch_size, seq_length, embed_size = self.x.shape

        # Gradients for W_O
        self.grad_W_O += torch.matmul(self.attention_out.view(-1, embed_size).t(), grad_output.view(-1, embed_size))  # [embed_size, embed_size]

        # Gradients for attention_out
        grad_attention_out = torch.matmul(grad_output, self.W_O.t())  # [batch_size, seq_length, embed_size]

        # Reshape to [batch_size, heads, seq_length, head_dim]
        grad_attention_out = grad_attention_out.view(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)  # [batch_size, heads, seq_length, head_dim]

        # Backward through attention
        grad_Q, grad_K, grad_V = self.attention.backward(grad_attention_out)  # Each is [batch_size, heads, seq_length, head_dim]

        # Compute gradients for W_Q, W_K, W_V
        # Reshape Q, K, V for gradient computation
        Q_flat = self.Q.contiguous().view(batch_size * self.heads * seq_length, self.head_dim)  # [batch_size * heads * seq_length, head_dim]
        K_flat = self.K.contiguous().view(batch_size * self.heads * seq_length, self.head_dim)  # [batch_size * heads * seq_length, head_dim]
        V_flat = self.V.contiguous().view(batch_size * self.heads * seq_length, self.head_dim)  # [batch_size * heads * seq_length, head_dim]

        # Reshape x for gradient computation
        x_flat = self.x.contiguous().view(batch_size * seq_length, embed_size)  # [batch_size * seq_length, embed_size]

        # Since W_Q maps embed_size to embed_size, sum gradients over heads
        # dL/dW_Q = x_flat.t() @ (grad_Q summed over heads)
        grad_Q_sum = grad_Q.sum(dim=1).contiguous().view(batch_size * seq_length, self.head_dim)  # [batch_size * seq_length, head_dim]
        self.grad_W_Q += torch.matmul(x_flat.t(), grad_Q_sum)  # [embed_size, head_dim]

        # Similarly for W_K and W_V
        grad_K_sum = grad_K.sum(dim=1).contiguous().view(batch_size * seq_length, self.head_dim)  # [batch_size * seq_length, head_dim]
        self.grad_W_K += torch.matmul(x_flat.t(), grad_K_sum)  # [embed_size, head_dim]

        grad_V_sum = grad_V.sum(dim=1).contiguous().view(batch_size * seq_length, self.head_dim)  # [batch_size * seq_length, head_dim]
        self.grad_W_V += torch.matmul(x_flat.t(), grad_V_sum)  # [embed_size, head_dim]

        # Gradients for x
        # Compute gradients from Q, K, V
        grad_x_Q = torch.matmul(grad_Q_sum, self.W_Q.t())  # [batch_size * seq_length, embed_size]
        grad_x_K = torch.matmul(grad_K_sum, self.W_K.t())  # [batch_size * seq_length, embed_size]
        grad_x_V = torch.matmul(grad_V_sum, self.W_V.t())  # [batch_size * seq_length, embed_size]

        # Total gradient for x
        grad_x = grad_x_Q + grad_x_K + grad_x_V  # [batch_size * seq_length, embed_size]

        # Reshape back to [batch_size, seq_length, embed_size]
        grad_x = grad_x.view(batch_size, seq_length, embed_size)  # [batch_size, seq_length, embed_size]
        print(f"MultiHeadAttention backward grad_output shape: {grad_output.shape}")

        return grad_x

    def zero_grad(self):
        """
        가중치의 그래디언트를 초기화합니다.
        """
        self.grad_W_Q.zero_()
        self.grad_W_K.zero_()
        self.grad_W_V.zero_()
        self.grad_W_O.zero_()
