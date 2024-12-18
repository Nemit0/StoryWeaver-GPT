import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer_torch import *
from src.tokenizer import BytePairTokenizer, load_tokenizer

def main():
    tokenizer_path = './model/tokenizer_shakesphere.json'
    model_path = './model/checkpoints/gpt_model_shakesphere.pth'

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer.token_map)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        embedding_dim = checkpoint.get('embedding_dim', 1024)
        max_seq_len = checkpoint.get('max_seq_len', 1024)
        heads = checkpoint.get('heads', 8)
        ff_expand_dim = checkpoint.get('ff_expand_dim', 2)
        blocks = checkpoint.get('blocks', 2)
        lr = checkpoint.get('lr', 0.001)

        model = GPTTorch(vocab_size=vocab_size, 
                    embed_size=embedding_dim, 
                    max_seq_len=max_seq_len, 
                    num_heads=heads, 
                    ff_expand=ff_expand_dim, 
                    num_blocks=blocks, 
                    dropout=0.1)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("Loaded model from file")
    else:
        raise Exception("Model not found")
    input = """ROMEO:
It is my soul that calls upon my name:
How silver-sweet sound lovers' tongues by night,
Like softest music to attending ears!

ROMEO:
My dear?"""
    output = generate_text_torch(model,
                        tokenizer,
                        input, max_tokens=512,
                        temperature=0.8,
                        frequency_penalty=0.2,
                        stop_on_repeat=False)
    print(output)

if __name__ == "__main__":
    main()