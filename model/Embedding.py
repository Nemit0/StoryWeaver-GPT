import torch
from torch import Tensor

### EMBEDDING ###
class Embedding:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Custom Embedding 레이어 초기화

        Args:
            input_dim (int): 임베딩할 인덱스의 개수 (예: 단어 집합의 크기)
            output_dim (int): 임베딩 벡터의 차원
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 임베딩 매트릭스를 학습 가능한 파라미터로 초기화
        self.weights: Tensor = torch.randn(input_dim, output_dim) * 0.01
        self.grad_weights: Tensor = torch.zeros_like(self.weights)

    def forward(self, input_indices: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            input_indices (Tensor): 정수 인덱스 텐서 (예: [batch_size, sequence_length])

        Returns:
            Tensor: 임베딩된 벡터 텐서 (예: [batch_size, sequence_length, output_dim])
        """
        self.input_indices = input_indices
        # 인덱스를 사용하여 임베딩 벡터 선택
        self.output = self.weights[input_indices]
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        """ 
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트 (예: [batch_size, sequence_length, output_dim])

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트 (임베딩 레이어의 경우 없음)
        """
        # grad_output의 형태: [batch_size, sequence_length, output_dim]
        # 이를 [batch_size * sequence_length, output_dim]로 평탄화
        grad_flat = grad_output.view(-1, self.output_dim)
        # input_indices를 평탄화하여 [batch_size * sequence_length] 형태로 
        input_flat = self.input_indices.view(-1)
        
        # 그래디언트를 초기화
        self.grad_weights.zero_()
        # 그래디언트 누적
        self.grad_weights.index_add_(0, input_flat, grad_flat)
        
        return None

    def __str__(self) -> str:
        return "CustomEmbedding"
