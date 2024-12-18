import os
import torch
import gradio as gr

from src.transformer import *
from src.transformer_torch import *
from src.tokenizer import *
from src.utils import *

model_path = "./model/checkpoints/test_model"
torch_model_path = "./model/checkpoints/gpt_model_shakesphere.pth"
tokenizer_path = "./model/tokenizer_shakesphere.json"

tokenizer = load_tokenizer(tokenizer_path)
model = GPT.load_model(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(torch_model_path):
    checkpoint = torch.load(torch_model_path, map_location=device)
    embedding_dim = checkpoint.get('embedding_dim', 1024)
    max_seq_len = checkpoint.get('max_seq_len', 1024)
    heads = checkpoint.get('heads', 8)
    ff_expand_dim = checkpoint.get('ff_expand_dim', 2)
    blocks = checkpoint.get('blocks', 2)
    lr = checkpoint.get('lr', 0.001)

    model_torch = GPTTorch(vocab_size=len(tokenizer.token_map),
                embed_size=embedding_dim, 
                max_seq_len=max_seq_len, 
                num_heads=heads, 
                ff_expand=ff_expand_dim, 
                num_blocks=blocks, 
                dropout=0.1).to(device)
    model_torch.load_state_dict(checkpoint['model_state_dict'])
else:
    import os
    print(os.listdir("./model/checkpoints"))
    raise Exception("Model not found")

def text_generation(input_text, temperature_value, repetition_penalty_value, model_name):
    if model_name == "Base Model":
        result = generate_sample_text(
            model,
            tokenizer,
            input_text,
            max_length=512,
            temperature=temperature_value,
            frequency_penalty=repetition_penalty_value
        )
    elif model_name == "Pytorch Model":
        result = generate_text_torch(
            model_torch,
            tokenizer,
            input_text,
            max_tokens=512,
            temperature=temperature_value,
            frequency_penalty=repetition_penalty_value
        )
    return result

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Simple Text Generation Playground")
        gr.Markdown("Enter your prompt, select a model, adjust temperature and repetition penalty, then press 'Generate'.")

        dropdown = ["Base Model", "Pytorch Model"]
        input_textbox = gr.Textbox(label="Input Text", lines=5, placeholder="Enter your prompt...")
        model_dropdown = gr.Dropdown(choices=dropdown, value="Base Model", label="Select Model")
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature")
        repetition_penalty_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="Repetition Penalty")
        generate_button = gr.Button("Generate")
        output_textbox = gr.Textbox(label="Output", lines=5)

        generate_button.click(
            fn=text_generation,
            inputs=[input_textbox, temperature_slider, repetition_penalty_slider, model_dropdown],
            outputs=output_textbox
        )

    demo.launch(share=True)

if __name__ == "__main__":
    main()