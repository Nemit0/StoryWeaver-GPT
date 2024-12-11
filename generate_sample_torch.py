import logging
import os

from src.transformer import *
from src.tokenizer import *
from src.utils import *

if torch.cuda.is_available():
    print("CUDA is available")
    torch.set_default_device('cuda')
else:
    print("CUDA is not available")
    torch.set_default_device('cpu')

tokenizer_path = './model/tokenizer_shakesphere.json'
model_path = './model/checkpoints/gpt_model_shakesphere.pth'
data_path = './data/input.txt'

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

    model = GPT(vocab_size=vocab_size, 
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

def generate_text(input: str, max_tokens: int = 100):
    input_tokens = tokenizer.encode(input)
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)
    input_tokens = input_tokens.to(torch.get_default_dtype())
    input_tokens = input_tokens.to(torch.get_default_device())
    model = GptObj.model
    model.eval()
    
    with torch.no_grad():
        for i in range(max_tokens):
            mask =  model.generate_causal_mask(input_tokens.size(1), torch.get_default_device())
            logits = model(input_tokens, mask)
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_tokens = torch.cat([input_tokens, next_token], dim=-1)
            if next_token == tokenizer.token_map['<EOS>']:
                break
    return input_tokens

def main():    
    # Load data
    with open(os.path.join(data_path), "r", encoding="utf-8") as f:
        text = f.read()

    sample_text = text[:100]
    
    # Generate text based on a sample input
    generated_tokens = generate_text(sample_text, max_tokens=100)
    
    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())
    
    # Print the original and generated text
    print("Sample input text: ")
    print(sample_text)
    print("\nGenerated text: ")
    print(generated_text)

if __name__ == "__main__":
    main()