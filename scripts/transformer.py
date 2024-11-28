import json
import re
import os
import torch
from torch import tensor, Tensor
from typing import List, Dict, Tuple
from multiprocessing import Pool
from tqdm import tqdm
from tokenizer import BytePairTokenizer, load_tokenizer

### CUDA SETUP ###

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_device(device)
    print(f"Using {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    torch.set_default_device(device)
    print("Using CPU")

### NEURAL NETWORK OBJECTS ###

class LinearLayer:
    def __init__(self, input_size: int, output_size: int):
        self.weights = torch.randn(input_size, output_size) * 0.01
        self.bias = torch.zeros(output_size)
        self.grad_weights = torch.zeros_like(self.weights)
        self.grad_bias = torch.zeros_like(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: Tensor) -> Tensor:
        self.grad_weights += self.input.transpose(0, 1) @ grad_output
        self.grad_bias += grad_output.sum(dim=0)
        grad_input = grad_output @ self.weights.transpose(0, 1)
        return grad_input

class ReLUActivation:
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return torch.clamp(x, min=0)

    def backward(self, grad_output: Tensor) -> Tensor:
        grad_input = grad_output.clone()
        grad_input[self.input <= 0] = 0
        return grad_input

class Embedding:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights: Tensor = torch.randn(input_dim, output_dim) * 0.01
        self.grad_weights: Tensor = torch.zeros_like(self.weights)

    def forward(self, input_indices: Tensor) -> Tensor:
        self.input_indices = input_indices
        self.output = self.weights[input_indices]
        return self.output

    def backward(self, grad_output: Tensor) -> None:
        grad_flat = grad_output.view(-1, self.output_dim)
        input_flat = self.input_indices.view(-1)
        # Accumulate gradients for embedding weights
        self.grad_weights.index_add_(0, input_flat, grad_flat)

class PositionalEncoding:
    def __init__(self, max_seq_len: int, embed_size: int):
        self.embed_size = embed_size
        self.pos_encoding = torch.zeros(max_seq_len, embed_size)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-torch.log(torch.tensor(10000.0)) / embed_size))
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x: Tensor) -> Tensor:
        seq_length, embed_size = x.shape
        pos_encoding = self.pos_encoding[:seq_length, :]  # Slice for the current sequence length
        return x + pos_encoding.to(x.device)

class MultiHeadAttention:
    def __init__(self, embed_size: int, heads: int):
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        if embed_size % heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads")

        # Initialize weights for each head
        self.W_Q = [torch.randn(embed_size, self.head_dim) * 0.01 for _ in range(heads)]
        self.W_K = [torch.randn(embed_size, self.head_dim) * 0.01 for _ in range(heads)]
        self.W_V = [torch.randn(embed_size, self.head_dim) * 0.01 for _ in range(heads)]
        self.W_O = [torch.randn(self.head_dim, embed_size) * 0.01 for _ in range(heads)]

        # Initialize gradients
        self.grad_W_Q = [torch.zeros_like(w) for w in self.W_Q]
        self.grad_W_K = [torch.zeros_like(w) for w in self.W_K]
        self.grad_W_V = [torch.zeros_like(w) for w in self.W_V]
        self.grad_W_O = [torch.zeros_like(w) for w in self.W_O]

    def forward(self, x: Tensor) -> Tensor:
        self.x = x  # Save for backward
        seq_length, _ = x.size()
        self.Q_heads, self.K_heads, self.V_heads = [], [], []
        for i in range(self.heads):
            Q = x @ self.W_Q[i]  # [seq_length, head_dim]
            K = x @ self.W_K[i]  # [seq_length, head_dim]
            V = x @ self.W_V[i]  # [seq_length, head_dim]
            self.Q_heads.append(Q)
            self.K_heads.append(K)
            self.V_heads.append(V)

        # Create mask for causal attention
        mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)
        self.head_outputs = []
        self.attention_weights = []
        self.scores = []
        for i in range(self.heads):
            scores = self.Q_heads[i] @ self.K_heads[i].transpose(0, 1) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
            scores += mask
            attn_weights = torch.softmax(scores, dim=-1)
            self.attention_weights.append(attn_weights)
            self.scores.append(scores)
            attn_output = attn_weights @ self.V_heads[i]
            head_output = attn_output @ self.W_O[i]
            self.head_outputs.append(head_output)

        self.out = sum(self.head_outputs)
        return self.out

    def backward(self, grad_output: Tensor) -> Tensor:
        grad_x = torch.zeros_like(self.x)
        for i in range(self.heads):
            # Since outputs are summed, grad_output is the same for each head
            grad_head_output = grad_output  # Shape: (seq_len, embed_size)

            # Gradient w.r.t. W_O[i]
            attn_output = self.attention_weights[i] @ self.V_heads[i]  # (seq_len, head_dim)
            self.grad_W_O[i] += attn_output.transpose(0, 1) @ grad_head_output  # (head_dim, embed_size)

            # Gradient w.r.t. attn_output
            grad_attn_output = grad_head_output @ self.W_O[i].transpose(0, 1)  # (seq_len, head_dim)

            # Gradient w.r.t. attention weights and V_heads[i]
            grad_attn_weights = grad_attn_output @ self.V_heads[i].transpose(0, 1)  # (seq_len, seq_len)
            grad_V = self.attention_weights[i].transpose(0, 1) @ grad_attn_output  # Corrected line

            # Softmax backward
            attn_weights = self.attention_weights[i]
            grad_scores = attn_weights * (grad_attn_weights - (attn_weights * grad_attn_weights).sum(dim=-1, keepdim=True))

            # Scale gradient by the scaling factor used in forward pass
            scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
            grad_Q = grad_scores @ self.K_heads[i] * scale  # (seq_len, head_dim)
            grad_K = grad_scores.transpose(0, 1) @ self.Q_heads[i] * scale  # (seq_len, head_dim)

            # Gradients w.r.t. weights
            self.grad_W_Q[i] += self.x.transpose(0, 1) @ grad_Q  # (embed_size, head_dim)
            self.grad_W_K[i] += self.x.transpose(0, 1) @ grad_K  # (embed_size, head_dim)
            self.grad_W_V[i] += self.x.transpose(0, 1) @ grad_V  # (embed_size, head_dim)

            # Accumulate gradients w.r.t. input x
            grad_x += grad_Q @ self.W_Q[i].transpose(0, 1)
            grad_x += grad_K @ self.W_K[i].transpose(0, 1)
            grad_x += grad_V @ self.W_V[i].transpose(0, 1)

        return grad_x

class AttentionBlock:
    def __init__(self, embed_size: int, heads: int):
        self.attention = MultiHeadAttention(embed_size, heads)
        self.layer_norm = LayerNorm(embed_size)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x  # Save input for backward
        attention_out = self.attention.forward(x)
        out = self.layer_norm.forward(attention_out + x)  # Residual connection
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        # Backward through layer norm
        grad_norm = self.layer_norm.backward(grad_output)

        # Residual connection
        grad_attention = grad_norm.clone()
        grad_x = grad_norm.clone()

        # Backward through attention
        grad_attention = self.attention.backward(grad_attention)

        # Add gradients from residual connection
        grad_x += grad_attention

        return grad_x

class LayerNorm:
    def __init__(self, embed_size: int, eps: float = 1e-5):
        self.gamma = torch.ones(embed_size)
        self.beta = torch.zeros(embed_size)
        self.grad_gamma = torch.zeros_like(self.gamma)
        self.grad_beta = torch.zeros_like(self.beta)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        self.mean = x.mean(dim=-1, keepdim=True)
        self.var = x.var(dim=-1, unbiased=False, keepdim=True)
        self.std = torch.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) / self.std
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        N, D = self.x.shape
        x_mu = self.x - self.mean
        std_inv = 1.0 / self.std

        # Gradients w.r.t. gamma and beta
        self.grad_gamma += torch.sum(grad_output * self.x_hat, dim=0)
        self.grad_beta += torch.sum(grad_output, dim=0)

        # Gradient w.r.t. x_hat
        dx_hat = grad_output * self.gamma

        # Gradient w.r.t. variance
        dvar = torch.sum(dx_hat * x_mu * -0.5 * std_inv.pow(3), dim=-1, keepdim=True)

        # Gradient w.r.t. mean
        dmu = torch.sum(-dx_hat * std_inv, dim=-1, keepdim=True) + dvar * torch.mean(-2.0 * x_mu, dim=-1, keepdim=True)

        # Gradient w.r.t. x
        dx = (dx_hat * std_inv) + (dvar * 2 * x_mu / N) + (dmu / N)
        return dx

class FeedForward:
    def __init__(self, embed_size: int, forward_expansion: int):
        self.fc1 = LinearLayer(embed_size, embed_size * forward_expansion)
        self.activation = ReLUActivation()
        self.fc2 = LinearLayer(embed_size * forward_expansion, embed_size)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x  # Save input for backward
        out = self.fc1.forward(x)
        out = self.activation.forward(out)
        out = self.fc2.forward(out)
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        grad = self.fc2.backward(grad_output)
        grad = self.activation.backward(grad)
        grad = self.fc1.backward(grad)
        return grad

class OutputProjection:
    def __init__(self, embed_size: int, vocab_size: int):
        self.W = torch.randn(embed_size, vocab_size) * 0.01
        self.grad_W = torch.zeros_like(self.W)

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        self.logits = x @ self.W
        return self.logits

    def backward(self, grad_output: Tensor) -> Tensor:
        self.grad_W += self.input.transpose(0, 1) @ grad_output
        grad_input = grad_output @ self.W.transpose(0, 1)
        return grad_input

class TransformerEncoderBlock:
    def __init__(self, embed_size: int, heads: int, ff_expand_dim: int):
        self.attention = AttentionBlock(embed_size, heads)
        self.layer_norm_1 = LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_expand_dim)
        self.layer_norm_2 = LayerNorm(embed_size)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x  # Save input for backward
        attention_out = self.attention.forward(x)
        x_residual = x + attention_out  # Residual connection
        x_norm = self.layer_norm_1.forward(x_residual)
        feed_forward_out = self.feed_forward.forward(x_norm)
        x_ff_residual = x_norm + feed_forward_out  # Residual connection
        out = self.layer_norm_2.forward(x_ff_residual)
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        # Backward through layer norm 2
        grad_norm_2 = self.layer_norm_2.backward(grad_output)

        # Residual connection with feed_forward_out
        grad_feed_forward = grad_norm_2.clone()
        grad_x_norm = grad_norm_2.clone()

        # Backward through feed forward network
        grad_feed_forward = self.feed_forward.backward(grad_feed_forward)

        # Add gradients from residual connection
        grad_x_norm += grad_feed_forward

        # Backward through layer norm 1
        grad_norm_1 = self.layer_norm_1.backward(grad_x_norm)

        # Residual connection with attention_out
        grad_attention = grad_norm_1.clone()
        grad_x = grad_norm_1.clone()

        # Backward through attention block
        grad_attention = self.attention.backward(grad_attention)

        # Add gradients from residual connection
        grad_x += grad_attention

        return grad_x

class GPT:
    def __init__(self, vocab_size: int, embed_size: int, max_seq_len: int, heads: int, ff_dim: int, num_blocks: int):
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks
        self.token_embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(max_seq_len, embed_size)
        self.transformer_blocks = []
        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerEncoderBlock(embed_size, heads, ff_dim))
        self.output = OutputProjection(embed_size, vocab_size)
        self.train: bool = True
        self.param_count = embed_size * vocab_size + embed_size * max_seq_len + embed_size * embed_size * 4 * num_blocks + embed_size * vocab_size

    def forward(self, x: Tensor, temperature: float = 1.0) -> Tensor:
        self.input_indices = x
        x = self.token_embedding.forward(x)
        x = self.positional_encoding.forward(x)
        for block in self.transformer_blocks:
            x = block.forward(x)
        logits = self.output.forward(x)
        self.logits = logits / temperature  # Save logits for backward
        probs = torch.softmax(self.logits, dim=-1)
        return probs

    def backward(self, probs: Tensor, labels: Tensor) -> None:
        # Compute gradient of loss w.r.t. logits
        # Assuming labels are indices of the target tokens
        batch_size, vocab_size = probs.shape
        labels_one_hot = torch.zeros_like(probs)
        labels_one_hot[range(batch_size), labels] = 1

        grad_logits = (probs - labels_one_hot) / batch_size  # Gradient w.r.t. logits

        # Backpropagate through output projection
        grad_output = self.output.backward(grad_logits)

        # Backpropagate through Transformer blocks
        for block in reversed(self.transformer_blocks):
            grad_output = block.backward(grad_output)

        # Backpropagate through positional encoding (no parameters)
        # Since positional encoding is additive and has no parameters, we can skip backward here

        # Backpropagate through token embedding
        self.token_embedding.backward(grad_output)

    def update_parameters(self, learning_rate: float):
        # Update token embedding weights
        self.token_embedding.weights -= learning_rate * self.token_embedding.grad_weights

        # Update output projection weights
        self.output.W -= learning_rate * self.output.grad_W

        # Update transformer blocks parameters
        for block in self.transformer_blocks:
            # Update attention parameters
            attention = block.attention.attention
            for i in range(attention.heads):
                attention.W_Q[i] -= learning_rate * attention.grad_W_Q[i]
                attention.W_K[i] -= learning_rate * attention.grad_W_K[i]
                attention.W_V[i] -= learning_rate * attention.grad_W_V[i]
                attention.W_O[i] -= learning_rate * attention.grad_W_O[i]

            # Update layer normalization parameters
            block.attention.layer_norm.gamma -= learning_rate * block.attention.layer_norm.grad_gamma
            block.attention.layer_norm.beta -= learning_rate * block.attention.layer_norm.grad_beta

            block.layer_norm_1.gamma -= learning_rate * block.layer_norm_1.grad_gamma
            block.layer_norm_1.beta -= learning_rate * block.layer_norm_1.grad_beta

            block.layer_norm_2.gamma -= learning_rate * block.layer_norm_2.grad_gamma
            block.layer_norm_2.beta -= learning_rate * block.layer_norm_2.grad_beta

            # Update feed forward parameters
            block.feed_forward.fc1.weights -= learning_rate * block.feed_forward.fc1.grad_weights
            block.feed_forward.fc1.bias -= learning_rate * block.feed_forward.fc1.grad_bias

            block.feed_forward.fc2.weights -= learning_rate * block.feed_forward.fc2.grad_weights
            block.feed_forward.fc2.bias -= learning_rate * block.feed_forward.fc2.grad_bias

    def zero_grad(self):
        # Zero gradients in token embedding
        self.token_embedding.grad_weights.zero_()

        # Zero gradients in output projection
        self.output.grad_W.zero_()

        # Zero gradients in transformer blocks
        for block in self.transformer_blocks:
            # Zero attention gradients
            attention = block.attention.attention
            for i in range(attention.heads):
                attention.grad_W_Q[i].zero_()
                attention.grad_W_K[i].zero_()
                attention.grad_W_V[i].zero_()
                attention.grad_W_O[i].zero_()

            # Zero layer normalization gradients
            block.attention.layer_norm.grad_gamma.zero_()
            block.attention.layer_norm.grad_beta.zero_()

            block.layer_norm_1.grad_gamma.zero_()
            block.layer_norm_1.grad_beta.zero_()

            block.layer_norm_2.grad_gamma.zero_()
            block.layer_norm_2.grad_beta.zero_()

            # Zero feed forward gradients
            block.feed_forward.fc1.grad_weights.zero_()
            block.feed_forward.fc1.grad_bias.zero_()

            block.feed_forward.fc2.grad_weights.zero_()
            block.feed_forward.fc2.grad_bias.zero_()

def main() -> None:
    # Set random seed
    torch.manual_seed(42)
    tokenizer = load_tokenizer()
    text = 'Hello World!'
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    vocab_size = len(tokenizer.token_map)
    embedding_dim = 64  # Reduced size for demonstration
    max_seq_len = 20  # Reduced sequence length for demonstration
    heads = 4
    ff_expand_dim = 4
    input_indices = torch.tensor(encoded)

    # Create GPT model
    Gpt_Object = GPT(vocab_size, embedding_dim, max_seq_len, heads, ff_expand_dim, num_blocks=2)
    Gpt_Object.train = True

    # Adjust input_indices and labels
    labels = input_indices[1:]  # Shifted input indices (length 5)
    input_indices = input_indices[:-1]  # Adjusted input indices (length 5)

    # Forward pass
    probs = Gpt_Object.forward(input_indices)
    predicted_tokens = torch.argmax(probs, dim=-1)
    print(f"Predicted Tokens: {predicted_tokens.tolist()}")

    # Compute loss (cross-entropy)
    loss = -torch.log(probs[range(len(labels)), labels]).mean()
    print(f"Loss before update: {loss.item()}")

    # Backward pass
    Gpt_Object.backward(probs, labels)

    # Perform a single parameter update
    learning_rate = 0.001
    Gpt_Object.update_parameters(learning_rate)

    # Zero gradients
    Gpt_Object.zero_grad()

    # Forward pass after update
    probs_after_update = Gpt_Object.forward(input_indices)
    predicted_tokens_after_update = torch.argmax(probs_after_update, dim=-1)
    print(f"Predicted Tokens after update: {predicted_tokens_after_update.tolist()}")

    # Compute loss after update
    loss_after_update = -torch.log(probs_after_update[range(len(labels)), labels]).mean()
    print(f"Loss after update: {loss_after_update.item()}")


if __name__ == "__main__":
    main()