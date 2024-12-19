import json
import os
import torch
import random
from src.torch_config import *
from src.transformer_batch import GPT
from src.tokenizer import BytePairTokenizer, load_tokenizer

def data_loader(path: str,
                       tokenizer,
                       batch_size: int,
                       seq_length: int,
                       vocab_size: int,
                       shuffle: bool = True):
    # Step 1: Read and encode the text
    with open(path, "r") as f:
        text = f.read()
    data = tokenizer.encode(text)
    
    # Step 2: Create sequences of length seq_length
    sequences = []
    for i in range(0, len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
        
    # Convert the list of sequences to a torch tensor
    sequences = torch.tensor(sequences, dtype=torch.long, device='cpu')
    
    # Step 3: Create a list of indices
    indices = list(range(len(sequences)))
    
    # Shuffle indices if required
    if shuffle:
        random.shuffle(indices)
    
    # Step 4: Create a generator that yields batches
    def batch_generator():
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx+batch_size]
            batch_data = sequences[batch_indices]
            yield batch_data
    
    # Return the actual generator object so it can be iterated over directly
    return batch_generator()


# def data_loader(path: str,
#                 tokenizer: BytePairTokenizer,
#                 batch_size: int,
#                 seq_length: int,
#                 vocab_size: int):
#     """
#     Data loader for training the model
#     """
#     with open(path, "r") as f:
#         text = f.read()
#     data = tokenizer.encode(text)
#     # Prepare sequences of length seq_length
#     sequences = []
#     for i in range(0, len(data) - seq_length):
#         seq = data[i:i+seq_length]
#         sequences.append(seq)
#     # Convert to tensor
#     sequences = torch.tensor(sequences)
#     # Create DataLoader
#     dataset = torch.utils.data.TensorDataset(sequences)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return loader

def main():
    tokenizer_path = "./model/tokenizer_shakesphere.json"
    model_path = "./model/checkpoints/batch_model"
    config_path = "./logs/config_batch.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            train_config = json.load(f)
    else:
        train_config = {
            "epochs": 0,
            "loss": []
        }
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer.token_map)
    embed_size = 768
    max_seq_len = 1024
    num_heads = 8
    forward_expansion = 2
    num_layers = 2
    lr = 1e-3
    model = GPT(
        vocab_size,
        embed_size,
        max_seq_len,
        num_heads,
        forward_expansion,
        num_layers,
        lr
    )
    batch_size = 16
    seq_length = max_seq_len + 1  # Adjust according to your needs
    path = "./data/input.txt"
    data_iter = data_loader(path, tokenizer, batch_size, seq_length, vocab_size)
    epochs = 5
    model.train_model(data_iter, epochs, batch_size)
    train_config["epochs"] += epochs
    # Assuming model.train_model returns a loss history
    # train_config["loss"].extend(loss)  # Update this line if needed
    with open(config_path, "w") as f:
        json.dump(train_config, f)
    model.save_model(model_path)

if __name__ == "__main__":
    main()