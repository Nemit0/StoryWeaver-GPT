import logging
import json
import sys
from datetime import datetime

from src.transformer import *
from src.tokenizer import *
from src.nn_objects import *
from src.utils import *

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

if torch.cuda.is_available():
    print("CUDA is available")
    torch.set_default_device('cuda')
    logging.log(logging.INFO, f"CUDA is available, using defice {torch.cuda.get_device_name()}")
else:
    print("CUDA is not available")
    torch.set_default_device('cpu')
    logging.log(logging.INFO, "CUDA is not available")

def main():
    tokenizer = load_tokenizer('./model/tokenizer_shakesphere.json')
    vocab_size = len(tokenizer.token_map)
    logging.info("Loaded tokenizer with vocab size %d", vocab_size)

    if os.path.exists('./logs/train_config.json'):
        with open('./logs/train_config.json', 'r') as f:
            train_config = json.load(f)
        logging.info(f"Loaded training config from file, picking up from epoch {train_config['epochs']}")
    else:
        train_config = {
            'epochs': 0,
            'loss': []
        }

    if os.path.exists('./model/gpt_model_shakesphere1.pth'):
        GptObj = GPT.load_model('./model/gpt_model_shakesphere1.pth')
        logging.log(logging.INFO, "Loaded model from file")
        embedding_dim = GptObj.embed_size
        max_seq_len = GptObj.max_seq_len
        heads = GptObj.heads
        ff_expand_dim = GptObj.ff_dim
    else:
        embedding_dim = 512
        max_seq_len = 512
        heads = 8
        ff_expand_dim = 2
        logging.log(logging.INFO, "Creating new model")
        GptObj = GPT(vocab_size, embedding_dim, max_seq_len, heads, ff_expand_dim, num_blocks=3)
    
    GptObj.train_mode = True

    # Load data
    with open(os.path.join(os.getcwd(), "./.data/input.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    data = tokenizer.encode(text)
    dataset = [torch.tensor(data[i:i+max_seq_len+1]) for i in range(0, len(data)-max_seq_len, max_seq_len)]
    print(len(dataset))

    dataset = dataset
    print(len(dataset))

    # Train the model
    epochs = 500
    learning_rate = 0.001
    logging.log(logging.INFO, "Training model for %d epochs with learning rate %f", epochs + train_config['epochs'], learning_rate)
    loss_history = GptObj.train_model(dataset, epochs, learning_rate)

    print(f"Final loss: {loss_history[-1]}")
    logging.info("Final loss: %f", loss_history[-1])

    train_config['epochs'] += epochs
    train_config['loss'].extend(loss_history)

    # Save the model
    GptObj.save_model('./model/gpt_model_shakesphere1.pth')
    logging.info("Model saved to file")
    with open('./logs/train_config.json', 'w') as f:
        json.dump(train_config, f)

if __name__ == "__main__":
    # try:
    #     main()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     logging.error("An error occurred: %s", e)
    #     sys.exit(1)
    # sys.exit(0)
    main()