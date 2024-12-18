import logging
import json
import os
import random
from datetime import datetime

from src.transformer_torch import *
from src.tokenizer import *
from src.utils import *

def main():
    print("Starting training...")

    # Configure logging
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    log_filename = f"./logs/{datetime.now().strftime('%Y%m%d%H%M%S')}_transformer.log"
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    print(f"Logging to file {log_filename}")

    # File paths
    tokenizer_path = './model/tokenizer_shakesphere.json'
    model_path = './model/checkpoints/gpt_model_shakesphere.pth'
    data_path = './data/input.txt'
    config_path = './logs/config_shakesphere.json'

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer.token_map)
    logging.info("Loaded tokenizer with vocab size %d", vocab_size)

    # Load training config if exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            train_config = json.load(f)
        logging.info("Loaded training config from file, starting from epoch %d", train_config['epochs'])
    else:
        train_config = {
            'epochs': 0,
            'loss': []
        }

    # If model exists, load it; otherwise create a new one
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
        embedding_dim = 1024
        max_seq_len = 1024
        heads = 8
        ff_expand_dim = 2
        lr = 0.001
        blocks = 2
        logging.info("Creating new model")
        model = GPT(vocab_size=vocab_size, 
                    embed_size=embedding_dim, 
                    max_seq_len=max_seq_len, 
                    num_heads=heads, 
                    ff_expand=ff_expand_dim, 
                    num_blocks=blocks, 
                    dropout=0.1)
    
    model = model.to(device)
    model.train()

    # Load data
    with open(os.path.join(data_path), "r", encoding="utf-8") as f:
        text = f.read()

    data = tokenizer.encode(text)
    dataset = [torch.tensor(data[i:i+max_seq_len+1], dtype=torch.long) for i in range(0, len(data)-max_seq_len, max_seq_len)]

    print(f"Dataset size: {len(dataset)}")
    logging.info("Dataset size: %d", len(dataset))

    # Train the model
    epochs = 500
    logging.info("Current epoch: %d, Training for %d more epochs with learning rate %f",
                 train_config['epochs'], epochs, lr)

    loss_history = train_model(model, dataset, epochs=epochs, lr=lr, device=device)

    print(f"Final loss: {loss_history[-1]}")
    logging.info("Final loss: %f", loss_history[-1])

    train_config['epochs'] += epochs
    train_config['loss'].extend(loss_history)

    # Save the model state and config
    os.makedirs('./model/checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'max_seq_len': max_seq_len,
        'heads': heads,
        'ff_expand_dim': ff_expand_dim,
        'blocks': blocks,
        'lr': lr
    }, model_path)
    logging.info("Model saved to %s", model_path)

    with open(config_path, 'w') as f:
        json.dump(train_config, f)
    logging.info("Training config saved to %s", config_path)

    # Generate a random sample from the dataset and produce sample tokens
    model.eval()
    random_sample = random.choice(dataset)
    initial_input = random_sample[:10]  # Take first 10 tokens from the random sample
    logging.info("Initial input tokens: %s", initial_input.tolist())
    print(f"Initial input tokens: {initial_input.tolist()}")

    generated_sequence = generate_sequence(model, initial_input, max_length=768, device=device)
    logging.info("Generated token sequence: %s", generated_sequence.tolist())
    print(f"Generated token sequence: {generated_sequence.tolist()}")

    decoded_initial = tokenizer.decode(initial_input.tolist())
    decoded_generated = tokenizer.decode(generated_sequence.tolist())
    logging.info("Decoded initial text: %s", decoded_initial)
    logging.info("Decoded generated text: %s", decoded_generated)
    print(f"Decoded initial text: {decoded_initial}")
    print(f"Decoded generated text: {decoded_generated}")

if __name__ == "__main__":
    main()