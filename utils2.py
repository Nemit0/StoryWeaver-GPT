import os
import torch
from typing import Callable
from datetime import datetime

def get_project_root(project_name:str) -> str:
    file_path = os.path.abspath(__file__)
    while os.path.basename(file_path) != project_name:
        file_path = os.path.dirname(file_path)
    return file_path

def timer(func:Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        print(f"Function {func.__name__} executed in {end_time - start_time}")
        return result
    return wrapper

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    디코더 마스크 생성

    Args:
        sz (int): 시퀀스 길이

    Returns:
        torch.Tensor: 마스크 텐서 [1, 1, sz, sz]
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, sz, sz]