import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional
from tqdm import tqdm

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
        pos_encoding = self.pe[:seq_len, :].unsqueeze(1).to(x.device)
        return x + pos_encoding

class GPTBlock(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, ff_expand: int, dropout: float = 0.1):
        super(GPTBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.attn = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=False)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_expand * embed_size),
            nn.GELU(),
            nn.Linear(ff_expand * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (seq_len, batch_size, embed_size)
        x_norm = self.ln1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attn_output)

        x_norm = self.ln2(x)
        ff_output = self.ff(x_norm)
        x = x + self.dropout(ff_output)
        return x

class GPT(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embed_size: int = 512,     
                 max_seq_len: int = 128, 
                 num_heads: int = 8, 
                 ff_expand: int = 4, 
                 num_blocks: int = 6, 
                 dropout: float = 0.1):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len=max_seq_len)
        self.blocks = nn.ModuleList([
            GPTBlock(embed_size, num_heads, ff_expand, dropout=dropout) for _ in range(num_blocks)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.output_proj = nn.Linear(embed_size, vocab_size)

        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # Causal mask: positions can only attend to previous positions
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

def train_model(model: nn.Module, 
                data: List[torch.Tensor], 
                epochs: int, 
                lr: float = 1e-3, 
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[float]:

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_history = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for batch in data:
            batch = batch.to(device)
            inputs = batch[:-1] # All but last
            targets = batch[1:] # All but first
            # Convert to batch dimension first
            # Bach isn't applied here yet
            inputs = inputs.unsqueeze(0)  # (1, seq_len)
            targets = targets.unsqueeze(0) # (1, seq_len)

            optimizer.zero_grad()
            seq_len = inputs.size(1)
            attn_mask = generate_causal_mask(seq_len, device=device)
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

def generate_sequence(model: nn.Module, initial_input: torch.Tensor, max_length: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    model.eval()
    initial_input = initial_input.to(device)
    input_seq = initial_input.unsqueeze(0)  # (1, seq_len)
    with torch.no_grad():
        for _ in range(max_length - initial_input.size(0)):
            seq_len = input_seq.size(1)
            attn_mask = generate_causal_mask(seq_len, device=device)
            logits = model(input_seq, attn_mask=attn_mask)  # (1, seq_len, vocab_size)
            next_token_probs = F.softmax(logits[:, -1, :], dim=-1)  
            next_token = torch.argmax(next_token_probs, dim=-1)  # greedy
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)

            if input_seq.size(1) > model.max_seq_len:
                input_seq = input_seq[:, -model.max_seq_len:]

    return input_seq.squeeze(0)