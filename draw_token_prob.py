from src.utils import *
from src.tokenizer import *
from src.transformer import *

def main():
    model_path = "./model/checkpoints/test_model"
    tokenizer_path = "./model/tokenizer_shakesphere.json"

    model = GPT.load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    input_text = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:"""
    temperature = 0.13
    frequency_penalty = 0.2
    max_length = 256
    greedy = False

    result = generate_sample_text(model, 
                                  tokenizer, 
                                  input_text, 
                                  max_length, 
                                  temperature, 
                                  frequency_penalty,
                                  greedy=greedy)
    print(result),

    visualize_prob(
        model,
        tokenizer,
        input_text,
        save_path="./visualize.gif",
        max_length=max_length,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        stop_token=None,
        greedy=greedy
    )

if __name__ == "__main__":
    main()