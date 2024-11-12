# feed_forward.py
import torch
from torch import Tensor
from nn_objects2 import Layer, Activation, Linear  # Linear 임포트

class FeedForward:
    def __init__(self, embed_size: int, forward_expansion: int, activation: Activation):
        """
        피드포워드 네트워크 초기화

        Args:
            embed_size (int): 임베딩 차원
            forward_expansion (int): 피드포워드 네트워크의 확장 비율
            activation (Activation): 활성화 함수
        """
        self.fc1 = Layer(embed_size, embed_size * forward_expansion, activation)
        self.fc2 = Layer(embed_size * forward_expansion, embed_size, Linear())  # Activation.Linear() -> Linear()로 수정

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [batch_size, seq_length, embed_size]

        Returns:
            Tensor: 피드포워드 네트워크 출력
        """
        out = self.fc1.forward(x)
        out = self.fc2.forward(out)
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트
        """
        grad = self.fc2.backward(grad_output)
        grad = self.fc1.backward(grad)
        return grad
