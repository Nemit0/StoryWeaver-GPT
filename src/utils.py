import torch
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Callable
from datetime import datetime

from .transformer import GPT
from .tokenizer import BytePairTokenizer
from .torch_config import *

def get_project_root(project_name: str) -> str:
    # function to get project root directory
    file_path = os.path.abspath(__file__)
    while os.path.basename(file_path) != project_name:
        file_path = os.path.dirname(file_path)
    return file_path

def timer(func: Callable) -> Callable:
    # decorator to measure execution time of a function
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        print(f"Function {func.__name__} executed in {end_time - start_time}")
        return result
    return wrapper

def generate_sample_text(
        model: GPT,
        tokenizer: BytePairTokenizer,
        input_text: str,
        max_length: int = 100,
        temperature: float = 1.0,
        frequency_penalty: float = 0.0,
        stop_token: str = None,
        greedy:bool = False) -> str:
    # function to generate text from a model
    input_tokens = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_tokens).to(device)

    stop_indices = tokenizer.encode(stop_token) if stop_token else None
    
    generated_tokens = model.generate_sequence(
        input_tensor, 
        max_length=max_length,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        stop_token=stop_token,
        greedy=greedy
    )

    generated_text = tokenizer.decode(generated_tokens.tolist())
    return generated_text

def single_step(
        model: GPT,
        input: torch.tensor,
        temperature: float = 1.0,
        freq_penalty_vector: torch.tensor = None,
        greedy:bool = False) -> torch.tensor:
    # performs a single step of token generation
    input = input.clone()
    probs = model.forward(input)
    next_token_probs = probs[-1] / temperature
    next_token_probs /= (1 + freq_penalty_vector)
    next_token_probs = torch.softmax(next_token_probs, dim=-1)
    if greedy:
        next_token = torch.argmax(next_token_probs)
    else:
        next_token = torch.multinomial(next_token_probs, 1)
    return next_token_probs, next_token

def visualize_prob(
        model: GPT,
        tokenizer: BytePairTokenizer,
        input_text: str,
        save_path: str,
        max_length: int = 10,
        temperature: float = 1.0,
        frequency_penalty: float = 0.0,
        stop_token: str = None,
        greedy: bool = False) -> None:
    # visualize probabilities of newly generated tokens
    # starting from after the initial input
    model.eval_mode()
    matplotlib.rcParams['animation.convert_path'] = '/usr/bin/convert'
    
    input_tokens = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_tokens)

    # store initial length to skip initial context in visualization
    initial_length = len(input_tensor)

    fig, (ax_text, ax_plot) = plt.subplots(1, 2, figsize=(8, 4))
    plt.tight_layout()
    ax_text.axis('off')

    def init():
        ax_text.text(0.5, 0.5, "", fontsize=14, ha='center', va='center')
        ax_plot.set_ylim(0, 1)
        ax_plot.set_xlabel("Token")
        ax_plot.set_ylabel("Probability")
        return []

    def update(step):
        nonlocal input_tensor
        ax_text.clear()
        ax_text.axis('off')

        # decode only the newly generated tokens after the initial input
        cumulative_text = tokenizer.decode(input_tensor[initial_length:initial_length+step].tolist())
        ax_text.text(0.5, 0.5, cumulative_text, fontsize=14, ha='center', va='center')

        ax_plot.clear()
        ax_plot.set_ylim(0, 1)
        ax_plot.set_xlabel("Token")
        ax_plot.set_ylabel("Probability")

        # always feed the entire sequence so far
        input_to_model = input_tensor
        next_token_probs, next_token = single_step(
            model, 
            input_to_model, 
            temperature=temperature, 
            freq_penalty_vector=torch.zeros_like(input_tensor[:1]),
            greedy=greedy
        )

        top_k_probs, top_k_indices = torch.topk(next_token_probs, 5)
        top_k_probs = top_k_probs.cpu().detach().numpy()
        top_k_indices = top_k_indices.cpu().detach().numpy()

        bars = ax_plot.bar(
            [tokenizer.decode([tid]) for tid in top_k_indices], 
            top_k_probs, 
            color='C0'
        )
        ax_plot.set_title(f"Step {step + 1}")

        # Reshape next_token to ensure it is a 1D tensor
        next_token = next_token.view(-1)
        
        # append newly generated token to input_tensor
        input_tensor = torch.cat([input_tensor, next_token], dim=0)

        return bars

    anim = animation.FuncAnimation(
        fig, 
        update, 
        init_func=init, 
        frames=max_length, 
        interval=1000, 
        blit=False
    )

    anim.save(save_path, writer='imagemagick', fps=1)
    plt.close(fig)