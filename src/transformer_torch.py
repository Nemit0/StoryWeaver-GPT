import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional
from tqdm import tqdm

from .tokenizer import *
from .torch_config import *

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pe = torch.zeros(max_len, embed_size, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch_size, embed_size)
        seq_len = x.size(0)
        pos_encoding = self.pe[:seq_len, :].unsqueeze(1)
        return x + pos_encoding

class GPTBlock(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, ff_expand: int, dropout: float = 0.1):
        super(GPTBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=False, device=device)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_expand * embed_size),
            nn.GELU(),
            nn.Linear(ff_expand * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (seq_len, batch_size, embed_size)
        if x.device != device:
            x = x.to(device)
        x_norm = self.ln1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_output)

        x_norm = self.ln2(x)
        ff_output = self.ff(x_norm)
        x = x + self.dropout(ff_output)
        return x

class GPTTorch(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embed_size: int = 512,     
                 max_seq_len: int = 128, 
                 num_heads: int = 8, 
                 ff_expand: int = 4, 
                 num_blocks: int = 6, 
                 dropout: float = 0.1):
        super(GPTTorch, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size, device=device)
        self.positional_encoding = PositionalEncoding(embed_size, max_len=max_seq_len)
        self.blocks = nn.ModuleList([
            GPTBlock(embed_size, num_heads, ff_expand, dropout=dropout) for _ in range(num_blocks)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.output_proj = nn.Linear(embed_size, vocab_size)

        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.type != device:
            x = x.to(device)
        # x shape: (batch_size, seq_len)
        # rearrange to (seq_len, batch_size)
        x = x.transpose(0, 1)  # -> (seq_len, batch_size)

        # Embedding
        x = self.token_embedding(x) * (self.embed_size ** 0.5)
        # Positional encoding
        x = self.positional_encoding(x)

        # Pass through GPT blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        # Output projection
        logits = self.output_proj(x)  # (seq_len, batch_size, vocab_size)
        return logits.transpose(0, 1)  # (batch_size, seq_len, vocab_size)

def generate_causal_mask(seq_len: int) -> torch.Tensor:
    # Causal mask: positions can only attend to previous positions
    mask = torch.full((seq_len, seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    return mask.to(device)

def train_model(model: nn.Module, 
                data: List[torch.Tensor], 
                epochs: int, 
                lr: float = 1e-3) -> List[float]:

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_history = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for batch in data:
            batch = batch
            inputs = batch[:-1] # All but last
            targets = batch[1:] # All but first
            # Convert to batch dimension first
            inputs = inputs.unsqueeze(0)  # (1, seq_len)
            targets = targets.unsqueeze(0) # (1, seq_len)

            optimizer.zero_grad()
            seq_len = inputs.size(1)
            attn_mask = generate_causal_mask(seq_len)
            logits = model(inputs, attn_mask=attn_mask)
            # logits shape: (1, seq_len, vocab_size), targets: (1, seq_len)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        if torch.isnan(torch.tensor(avg_loss)):
            print("Loss is NaN, stopping.")
            break

    return loss_history

def generate_sequence(model: nn.Module, 
                      initial_input: torch.Tensor, 
                      max_length: int, 
                      tokenizer: BytePairTokenizer,
                      temperature: float = 1.0,
                      frequency_penalty: float = 0.0,
                      stop_on_repeat: bool = True) -> torch.Tensor:
    model.eval()
    initial_input = initial_input.to(device)
    input_seq = initial_input.unsqueeze(0)  # (1, seq_len)
    text = tokenizer.decode(initial_input.tolist())
    token_frequencies = {}

    with torch.no_grad():
        for _ in range(max_length - initial_input.size(0)):
            seq_len = input_seq.size(1)
            attn_mask = generate_causal_mask(seq_len)
            logits = model(input_seq, attn_mask=attn_mask)  # (1, seq_len, vocab_size)
            logits = logits[:, -1, :] / temperature  # apply temperature

            # Apply frequency penalty
            for token, freq in token_frequencies.items():
                logits[:, token] -= frequency_penalty * freq

            next_token_probs = F.softmax(logits, dim=-1).squeeze(0)
            next_token = torch.multinomial(next_token_probs, 1)  # probabilistic sampling

            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)

            if input_seq.size(1) > model.max_seq_len:
                input_seq = input_seq[:, -model.max_seq_len:]

            next_token_id = next_token.item()
            text += tokenizer.decode([next_token_id])

            # Update token frequencies
            if next_token_id in token_frequencies:
                token_frequencies[next_token_id] += 1
            else:
                token_frequencies[next_token_id] = 1

            if stop_on_repeat and check_output_reproduction(text):
                break

    return input_seq.squeeze(0)

def check_output_reproduction(generated_text: str, repeat_threshold: int = 10, repeat_length: int = None) -> bool:
    if repeat_length is None:
        repeat_length = len(generated_text) // 10
    
    for i in range(len(generated_text) - repeat_length):
        for j in range(5, repeat_length):
            if generated_text[i:i+j] == generated_text[i+j:i+2*j]:
                if j >= repeat_threshold:
                    return True
    return False

def generate_text_torch(model: nn.Module,
                        tokenizer: BytePairTokenizer,
                        input: str, 
                        max_tokens: int = 100, 
                        temperature: float = 1.0, 
                        frequency_penalty: float = 0.0, 
                        stop_on_repeat: bool = True) -> str:
    input_tokens = tokenizer.encode(input)
    input_tokens = torch.tensor(input_tokens)
    model.eval()
    output_tokens = generate_sequence(model, 
                                      input_tokens, 
                                      max_tokens, 
                                      tokenizer=tokenizer,
                                      temperature=temperature,
                                      frequency_penalty=frequency_penalty,
                                      stop_on_repeat=stop_on_repeat)
    output_string = tokenizer.decode(output_tokens.tolist())
    return output_string