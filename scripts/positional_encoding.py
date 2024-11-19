# positional_encoding.py
import torch
from torch import Tensor
import math

class PositionalEncoding:
    def __init__(self, max_seq_len: int, embed_size: int):
        """
        위치 인코딩 초기화

        Args:
            max_seq_len (int): 최대 시퀀스 길이
            embed_size (int): 임베딩 차원
        """
        self.embed_size = embed_size
        self.pos_encoding = torch.zeros(max_seq_len, embed_size)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = self.pos_encoding.unsqueeze(0)  # [1, max_seq_len, embed_size]

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 임베딩된 입력 텐서 [batch_size, seq_length, embed_size]

        Returns:
            Tensor: 위치 인코딩이 추가된 텐서 [batch_size, seq_length, embed_size]
        """
        seq_length = x.size(1)
        return x + self.pos_encoding[:, :seq_length, :]