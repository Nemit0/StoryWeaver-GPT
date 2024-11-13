# simple_linear.py
import torch
from torch import Tensor

class SimpleLinear:
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Args:
            input_size (int): 입력 피처의 크기
            output_size (int): 출력 피처의 크기
        """
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.weights: Tensor = torch.rand(input_size, output_size) # 가중치 랜덤 초기화

    def forward(self, inputs: Tensor) -> Tensor:
        """
        입력에 가중치를 단순 행렬곱하여 출력

        Args:
            inputs (Tensor): 입력 텐서 [batch_size, input_size]

        Returns:
            Tensor: 출력 텐서 [batch_size, output_size]
        """
        self.inputs: Tensor = inputs
        self.output: Tensor = torch.mm(inputs, self.weights) # 단순 행렬곱
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        손실 함수 그래디언트 이전 층으로 전달 및 가중치 그래디언트 계산

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트 [batch_size, output_size]

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트 [batch_size, input_size]
        """

        grad_input: Tensor = torch.mm(grad_output, self.weights.t()) # 단순 행렬곱
        self.grad_weights: Tensor = torch.mm(self.inputs.t(), grad_output)
        return grad_input
