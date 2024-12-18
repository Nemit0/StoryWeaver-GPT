import logging
import json
import os
from datetime import datetime

from src.transformer import *
from src.tokenizer import *
from src.nn_objects import *
from src.utils import *

print("Starting training...")

# Configure logging
log_filename = f"./logs/{datetime.now().strftime('%Y%m%d%H%M%S')}_transformer.log"
logging.basicConfig(
    filename = log_filename,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if not os.path.exists('./logs'):
    os.makedirs('./logs')
print(f"Logging to file {log_filename}")

if torch.cuda.is_available():
    print("CUDA is available")
    torch.set_default_device('cuda')
    logging.log(logging.INFO, f"CUDA is available, using defice {torch.cuda.get_device_name()}")
else:
    print("CUDA is not available")
    torch.set_default_device('cpu')
    logging.log(logging.INFO, "CUDA is not available")

def main():
    print("Starting training...")
    # Make necessary directories for training
    # Some models are ignored in the .gitignore file
    # Hence it's necessary to create these directories
    if not os.path.exists('./model/checkpoints'):
        os.makedirs('./model/checkpoints')
    tokenizer_path = './model/tokenizer_shakesphere.json'
    model_path = './model/checkpoints/gpt_model'
    data_path = './data/input.txt'
    config_path = './logs/config_1.json'

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer.token_map)
    logging.info("Loaded tokenizer with vocab size %d", vocab_size)

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            train_config = json.load(f)
        logging.info(f"Loaded training config from file, picking up from epoch {train_config['epochs']}")
    else:
        train_config = {
            'epochs': 0,
            'loss': []
        }

    if os.path.exists(model_path):
        GptObj = GPT.load_model(model_path)
        logging.log(logging.INFO, "Loaded model from file")
        embedding_dim = GptObj.embed_size
        max_seq_len = GptObj.max_seq_len
        heads = GptObj.heads
        ff_expand_dim = GptObj.ff_dim
        lr = GptObj.lr
    else:
        embedding_dim =768
        max_seq_len = 512
        heads = 4
        ff_expand_dim = 2
        lr = 0.001
        blocks = 2
        logging.log(logging.INFO, "Creating new model")
        GptObj = GPT(vocab_size, 
                     embedding_dim, 
                     max_seq_len, 
                     heads, 
                     ff_expand_dim, 
                     num_blocks=blocks,
                     lr = lr)
    
    GptObj.train_mode = True

    # Load data
    with open(os.path.join(data_path), "r", encoding="utf-8") as f:
        text = f.read()
    
    data = tokenizer.encode(text)
    dataset = [torch.tensor(data[i:i+max_seq_len+1]) for i in range(0, len(data)-max_seq_len, int(max_seq_len))]

    # For demonstration, we can use a smaller dataset
    dataset = dataset
    print(len(dataset))
    print(f"Dataset size: {len(dataset)}")

    # Train the model
    epochs = 600
    logging.log(logging.INFO, f"Current epoch: {train_config['epochs']}, Training for {epochs} more epochs with learning rate {lr}")
    loss_history = GptObj.train_model(dataset, 
                                      epochs, 
                                      model_path=model_path, 
                                      config_path=config_path,)

    print(f"Final loss: {loss_history[-1]}")
    logging.info("Final loss: %f", loss_history[-1])

    train_config['epochs'] += epochs
    train_config['loss'].extend(loss_history)

    # Save the model
    GptObj.save_model(model_path)
    logging.info("Model saved to file")
    with open(config_path, 'w') as f:
        json.dump(train_config, f)

if __name__ == "__main__":
    main()