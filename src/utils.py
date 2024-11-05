import os

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