import torch
import os

from src.transformer import GPT
from src.tokenizer import load_tokenizer

def generate_sample_text(model_path: str, 
                         tokenizer_path: str, 
                         initial_text: str, max_length: int = 50
                         ) -> str:

    model:GPT = GPT.load_model(model_path)

    # Load the tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}. Please ensure the path is correct.")

    tokenizer = load_tokenizer(tokenizer_path)
    
    # Encode the initial input text into tensor of token indices
    input_tokens = tokenizer.encode(initial_text)
    input_tensor = torch.tensor(input_tokens)

    # Generate sequence
    generated_tokens = model.generate_sequence(input_tensor, 
                                               max_length=max_length,
                                               temperature=0.9)

    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(generated_tokens.tolist())
    
    return generated_text

if __name__ == "__main__":
    model_path = "./model/checkpoints/test_model"
    tokenizer_path = "./model/tokenizer_shakesphere.json"

    # Example initial input
    initial_text = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, """
    max_length = 512
    output = generate_sample_text(model_path, tokenizer_path, initial_text, max_length)
    print("Generated text:")
    print(output)