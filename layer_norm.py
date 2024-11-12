# layer_norm.py
import torch
from torch import Tensor

class LayerNorm:
    def __init__(self, embed_size: int, eps: float = 1e-5):
        """
        레이어 정규화 초기화

        Args:
            embed_size (int): 임베딩 차원
            eps (float, optional): 안정성을 위한 작은 값. Defaults to 1e-5.
        """
        self.gamma = torch.ones(embed_size, requires_grad=False)
        self.beta = torch.zeros(embed_size, requires_grad=False)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [batch_size, seq_length, embed_size]

        Returns:
            Tensor: 정규화된 텐서
        """
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True)
        self.normalized = (x - self.mean) / (self.std + self.eps)
        return self.gamma * self.normalized + self.beta

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트
        """
        # 단순화를 위해 역전파는 gamma에 대한 기울기만 처리
        return grad_output * self.gamma
