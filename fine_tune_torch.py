import torch
import json

from src.transformer_torch import *
from src.tokenizer import *

def main():
    base_model_path = './model/checkpoints/gpt_model_shakesphere.pth'
    fine_tuned_model_path = './model/checkpoints/gpt_model_finetuned.pth'
    tokenizer_path = './model/tokenizer_shakesphere.json'
    data_path = './data/finetune.json'
    config_path = './logs/config_finetune.json'

    tokenizer = load_tokenizer(tokenizer_path) 
    vocab_size = len(tokenizer.token_map)
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        raise FileNotFoundError("Data file not found")

    if os.path.exists(fine_tuned_model_path):
        checkpoint = torch.load(fine_tuned_model_path)
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
    else:
        checkpoint = torch.load(base_model_path)
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

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            train_config = json.load(f)
    else:
        train_config = {
            'epochs': 0,
            'loss': []
        }
    
    data = {
        "input": [line['question'] for line in data],
        "output": [line['answer'] for line in data]
    }
    
    epoch = 50

    model.train_mode = True
    dataset = InputOutputDataset(data['input'], data['output'], tokenizer, max_length=512)
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=collate_fn, 
        generator=torch.Generator(device=device) 
    )
    loss_history = train_finetune(model, dataloader, tokenizer, epochs=epoch, lr=1e-4)
    train_config['epochs'] += epoch
    train_config['loss'].extend(loss_history)
    with open(config_path, 'w') as f:
        json.dump(train_config, f)

    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'max_seq_len': max_seq_len,
        'heads': heads,
        'ff_expand_dim': ff_expand_dim,
        'blocks': blocks,
        'lr': lr
    }, fine_tuned_model_path)

if __name__ == '__main__':
    main()