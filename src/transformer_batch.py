import torch
import torch.nn.functional as F
import sys
import signal
import json
from tqdm import tqdm
from torch import Tensor
from typing import List, Dict, Tuple, Optional

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("Using CPU")

def clones(module, N):
    """Produce N identical layers."""
    return [module() for _ in range(N)]

class LinearLayer:
    def __init__(self, input_size: int, output_size: int, device=device):
        self.weights = torch.randn(input_size, output_size, dtype=torch.float32, device=device) * 0.01
        self.bias = torch.zeros(output_size, dtype=torch.float32, device=device)
        # Gradients
        self.grad_weights = torch.zeros_like(self.weights)
        self.grad_bias = torch.zeros_like(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        self.input = x  # Save for backward
        output = x @ self.weights + self.bias
        return output

    def backward(self, grad_output: Tensor) -> Tensor:
        # Compute gradients w.r.t weights and biases using einsum
        self.grad_weights += torch.einsum('bsi,bso->io', self.input, grad_output)
        self.grad_bias += grad_output.sum(dim=[0, 1])  # Sum over batch and sequence dimensions

        # Compute gradient w.r.t input
        grad_input = grad_output @ self.weights.transpose(0, 1)  # [batch_size, seq_length, input_size]

        return grad_input

    def zero_grad(self):
        self.grad_weights.zero_()
        self.grad_bias.zero_()

class LayerNorm:
    def __init__(self, embed_size: int, eps: float = 1e-5, device=device):
        self.gamma = torch.ones(embed_size, dtype=torch.float32, device=device)
        self.beta = torch.zeros(embed_size, dtype=torch.float32, device=device)
        self.eps = eps
        # Gradients
        self.grad_gamma = torch.zeros_like(self.gamma)
        self.grad_beta = torch.zeros_like(self.beta)

    def forward(self, x: Tensor) -> Tensor:
        self.input = x  # Save for backward
        self.mean = x.mean(dim=-1, keepdim=True)
        self.var = x.var(dim=-1, unbiased=False, keepdim=True)
        self.std = torch.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) / self.std
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        N, D = self.input.shape[-2], self.input.shape[-1]
        x_mu = self.input - self.mean
        std_inv = 1.0 / self.std

        # Gradients w.r.t. gamma and beta
        self.grad_gamma += (grad_output * self.x_hat).sum(dim=tuple(range(len(grad_output.shape)-1)))
        self.grad_beta += grad_output.sum(dim=tuple(range(len(grad_output.shape)-1)))

        # Gradient w.r.t. x_hat
        dx_hat = grad_output * self.gamma

        # Intermediate partial derivatives
        dvar = (-0.5 * std_inv**3 * (dx_hat * x_mu).sum(dim=-1, keepdim=True))
        dmean = (-std_inv * dx_hat).sum(dim=-1, keepdim=True) + dvar * (-2.0 * x_mu.mean(dim=-1, keepdim=True))

        # Gradient w.r.t. x
        grad_input = dx_hat * std_inv + dvar * 2 * x_mu / D + dmean / D
        return grad_input

    def zero_grad(self):
        self.grad_gamma.zero_()
        self.grad_beta.zero_()

class Embedding:
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.weights = torch.randn(vocab_size, embed_size, dtype=torch.float32, device=device) * 0.01
        # Gradient
        self.grad_weights = torch.zeros_like(self.weights)

    def forward(self, input_indices: Tensor) -> Tensor:
        # input_indices: [batch_size, seq_length]
        self.input_indices = input_indices  # Save for backward
        output = self.weights[input_indices]  # [batch_size, seq_length, embed_size]
        return output

    def backward(self, grad_output: Tensor) -> None:
        # grad_output: [batch_size, seq_length, embed_size]
        batch_size, seq_length, embed_size = grad_output.shape
        grad_flat = grad_output.view(-1, embed_size)  # [batch_size * seq_length, embed_size]
        input_flat = self.input_indices.reshape(-1)  # [batch_size * seq_length]
        # Accumulate gradients
        self.grad_weights.index_add_(0, input_flat, grad_flat)

    def zero_grad(self):
        self.grad_weights.zero_()

class PositionalEncoding:
    def __init__(self, max_seq_len: int, embed_size: int):
        self.embed_size = embed_size
        position = torch.arange(0, max_seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pe = torch.zeros(max_seq_len, embed_size, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_encoding = pe.unsqueeze(0)  # [1, max_seq_len, embed_size]

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_length, embed_size]
        seq_length = x.shape[1]
        x = x + self.pos_encoding[:, :seq_length, :]
        return x

# Activation Functions
class ReLUActivation:
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return torch.clamp(x, min=0)

    def backward(self, grad_output: Tensor) -> Tensor:
        grad_input = grad_output.clone()
        grad_input[self.input <= 0] = 0
        return grad_input

class GeLUActivation:
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        c = torch.sqrt(torch.tensor(2 / torch.pi, dtype=torch.float32, device=device))
        self.gelu = 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * x ** 3)))
        return self.gelu

    def backward(self, grad_output: Tensor) -> Tensor:
        x = self.input
        c = torch.sqrt(torch.tensor(2 / torch.pi, dtype=torch.float32, device=device))
        tanh_out = torch.tanh(c * (x + 0.044715 * x ** 3))
        sech2 = 1 - tanh_out ** 2
        term1 = 0.5 * (1 + tanh_out)
        term2 = (0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x ** 2))
        grad_input = grad_output * (term1 + term2)
        return grad_input

class MultiHeadAttention:
    def __init__(self, embed_size: int, num_heads: int, device=device):
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # We combine all the projections into single matrices for efficiency
        self.W_Q = torch.randn(embed_size, embed_size, dtype=torch.float32, device=device) * 0.01
        self.W_K = torch.randn(embed_size, embed_size, dtype=torch.float32, device=device) * 0.01
        self.W_V = torch.randn(embed_size, embed_size, dtype=torch.float32, device=device) * 0.01
        self.W_O = torch.randn(embed_size, embed_size, dtype=torch.float32, device=device) * 0.01

        # Gradients
        self.grad_W_Q = torch.zeros_like(self.W_Q)
        self.grad_W_K = torch.zeros_like(self.W_K)
        self.grad_W_V = torch.zeros_like(self.W_V)
        self.grad_W_O = torch.zeros_like(self.W_O)

    def split_heads(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_length, embed_size]
        batch_size, seq_length, embed_size = x.shape
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_length, head_dim]
        return x

    def combine_heads(self, x: Tensor) -> Tensor:
        # x: [batch_size, num_heads, seq_length, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_length, num_heads, head_dim]
        batch_size, seq_length, num_heads, head_dim = x.shape
        x = x.view(batch_size, seq_length, num_heads * head_dim)  # [batch_size, seq_length, embed_size]
        return x

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_length, embed_size]
        self.x = x  # Save for backward
        batch_size, seq_length, embed_size = x.shape

        # Linear projections
        Q = x @ self.W_Q  # [batch_size, seq_length, embed_size]
        K = x @ self.W_K
        V = x @ self.W_V

        # Split into heads
        Q = self.split_heads(Q)  # [batch_size, num_heads, seq_length, head_dim]
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Save K, V for backward
        self.K = K
        self.V = V
        self.Q = Q

        # Scaled dot-product attention
        dk = self.head_dim
        scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=device))
        # Create mask for causal attention
        mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        self.scores = scores  # Save for backward
        attn_weights = F.softmax(scores, dim=-1)
        self.attn_weights = attn_weights  # Save for backward

        attn_output = attn_weights @ V  # [batch_size, num_heads, seq_length, head_dim]

        # Combine heads
        attn_output = self.combine_heads(attn_output)  # [batch_size, seq_length, embed_size]

        # Final linear layer
        output = attn_output @ self.W_O  # [batch_size, seq_length, embed_size]
        self.attn_output = attn_output  # Save for backward
        return output

    def backward(self, grad_output: Tensor) -> Tensor:
        # grad_output: [batch_size, seq_length, embed_size]
        batch_size, seq_length, embed_size = grad_output.shape

        # Gradients w.r.t. W_O
        self.grad_W_O += self.attn_output.view(-1, embed_size).transpose(0, 1) @ grad_output.view(-1, embed_size)
        grad_attn_output = grad_output @ self.W_O.transpose(0, 1)  # [batch_size, seq_length, embed_size]

        # Reshape grad_attn_output to [batch_size, num_heads, seq_length, head_dim]
        grad_attn_output = self.split_heads(grad_attn_output)

        # Gradients w.r.t. attn_weights and V
        grad_attn_weights = grad_attn_output @ self.V.transpose(-2, -1)
        grad_V = self.attn_weights.transpose(-2, -1) @ grad_attn_output

        # Softmax backward
        d_scores = self.softmax_backward(grad_attn_weights, self.attn_weights)

        # Scale by scaling factor
        dk = self.head_dim
        scale = 1.0 / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=device))
        grad_Q = d_scores @ self.K
        grad_K = d_scores.transpose(-2, -1) @ self.Q

        grad_Q *= scale
        grad_K *= scale

        # Gradients w.r.t. W_Q, W_K, W_V
        grad_Q_combined = self.combine_heads(grad_Q)
        grad_K_combined = self.combine_heads(grad_K)
        grad_V_combined = self.combine_heads(grad_V)

        # Gradients w.r.t. input x
        self.grad_W_Q += self.x.view(-1, embed_size).transpose(0, 1) @ grad_Q_combined.view(-1, embed_size)
        self.grad_W_K += self.x.view(-1, embed_size).transpose(0, 1) @ grad_K_combined.view(-1, embed_size)
        self.grad_W_V += self.x.view(-1, embed_size).transpose(0, 1) @ grad_V_combined.view(-1, embed_size)

        grad_x_Q = grad_Q_combined @ self.W_Q.transpose(0, 1)
        grad_x_K = grad_K_combined @ self.W_K.transpose(0, 1)
        grad_x_V = grad_V_combined @ self.W_V.transpose(0, 1)

        grad_x = grad_x_Q + grad_x_K + grad_x_V
        return grad_x

    def softmax_backward(self, grad_output: Tensor, softmax_output: Tensor) -> Tensor:
        # Softmax backward for batch inputs
        # grad_output: [batch_size, num_heads, seq_length, seq_length]
        # softmax_output: [batch_size, num_heads, seq_length, seq_length]
        grad_input = grad_output.clone()
        sum_grad = (grad_output * softmax_output).sum(dim=-1, keepdim=True)
        grad_input = softmax_output * (grad_output - sum_grad)
        return grad_input

    def zero_grad(self):
        self.grad_W_Q.zero_()
        self.grad_W_K.zero_()
        self.grad_W_V.zero_()
        self.grad_W_O.zero_()

class FeedForward:
    def __init__(self, embed_size: int, forward_expansion: int, device=device):
        self.fc1 = LinearLayer(embed_size, embed_size * forward_expansion, device=device)
        self.activation = GeLUActivation()
        self.fc2 = LinearLayer(embed_size * forward_expansion, embed_size, device=device)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x  # Save for backward
        out = self.fc1.forward(x)
        out = self.activation.forward(out)
        out = self.fc2.forward(out)
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        grad = self.fc2.backward(grad_output)
        grad = self.activation.backward(grad)
        grad = self.fc1.backward(grad)
        return grad

    def zero_grad(self):
        self.fc1.zero_grad()
        self.fc2.zero_grad()

class TransformerEncoderBlock:
    def __init__(self, embed_size: int, num_heads: int, forward_expansion: int, device=device):
        self.attention = MultiHeadAttention(embed_size, num_heads, device=device)
        self.layer_norm1 = LayerNorm(embed_size, device=device)
        self.feed_forward = FeedForward(embed_size, forward_expansion, device=device)
        self.layer_norm2 = LayerNorm(embed_size, device=device)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x  # Save input for backward
        attn_out = self.attention.forward(x)
        x = x + attn_out  # Residual connection
        x = self.layer_norm1.forward(x)
        ff_out = self.feed_forward.forward(x)
        x = x + ff_out  # Residual connection
        x = self.layer_norm2.forward(x)
        return x

    def backward(self, grad_output: Tensor) -> Tensor:
        # Backward through layer_norm2
        grad = self.layer_norm2.backward(grad_output)
        # Residual connection with feed_forward output
        grad_ff = grad.clone()
        grad_x = grad.clone()
        grad_ff = self.feed_forward.backward(grad_ff)
        grad_x += grad_ff
        # Backward through layer_norm1
        grad = self.layer_norm1.backward(grad_x)
        # Residual connection with attention output
        grad_attn = grad.clone()
        grad_input = grad.clone()
        grad_attn = self.attention.backward(grad_attn)
        grad_input += grad_attn
        return grad_input

    def zero_grad(self):
        self.attention.zero_grad()
        self.layer_norm1.zero_grad()
        self.feed_forward.zero_grad()
        self.layer_norm2.zero_grad()

class OutputProjection:
    def __init__(self, embed_size: int, vocab_size: int):
        self.W = torch.randn(embed_size, vocab_size, dtype=torch.float32, device=device) * 0.01
        self.grad_W = torch.zeros_like(self.W)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_length, embed_size]
        self.input = x  # Save for backward
        logits = x @ self.W  # [batch_size, seq_length, vocab_size]
        return logits

    def backward(self, grad_output: Tensor) -> Tensor:
        # grad_output: [batch_size, seq_length, vocab_size]
        batch_size, seq_length, vocab_size = grad_output.shape
        x_reshaped = self.input.reshape(-1, self.W.shape[0])  # [(batch_size * seq_length), embed_size]
        grad_output_reshaped = grad_output.reshape(-1, vocab_size)  # [(batch_size * seq_length), vocab_size]
        self.grad_W += x_reshaped.transpose(0, 1) @ grad_output_reshaped  # [embed_size, vocab_size]
        grad_input = grad_output_reshaped @ self.W.transpose(0, 1)  # [(batch_size * seq_length), embed_size]
        grad_input = grad_input.view(batch_size, seq_length, -1)  # [batch_size, seq_length, embed_size]
        return grad_input

    def zero_grad(self):
        self.grad_W.zero_()

# Optimizer
class AdamOptimizer:
    def __init__(self, params: List[Tensor], grads: List[Tensor], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        """
        Initializes the Adam optimizer.
        Args:
            params (List[Tensor]): List of parameters to optimize.
            grads (List[Tensor]): List of corresponding gradients.
            lr (float): Learning rate.
            betas (Tuple[float, float]): Coefficients used for computing running averages.
            eps (float): Term added to the denominator to improve numerical stability.
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.params = params
        self.grads = grads
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0  # Time step

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, self.grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Update parameters
            param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        Resets all gradients to zero.
        """
        for grad in self.grads:
            grad.zero_()

class GPT:
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 max_seq_len: int,
                 num_heads: int,
                 forward_expansion: int,
                 num_layers: int,
                 lr: float = 1e-3):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.num_layers = num_layers
        self.lr = lr

        # Layers
        self.embedding = Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_size)
        self.layers = [TransformerEncoderBlock(embed_size, num_heads, forward_expansion) for _ in range(num_layers)]
        self.output_proj = OutputProjection(embed_size, vocab_size)

        # Optimizer parameters
        self.params = [self.embedding.weights,
                       self.output_proj.W]
        self.grads = [self.embedding.grad_weights,
                      self.output_proj.grad_W]

        for layer in self.layers:
            # Attention parameters
            self.params.extend([layer.attention.W_Q,
                                layer.attention.W_K,
                                layer.attention.W_V,
                                layer.attention.W_O])
            self.grads.extend([layer.attention.grad_W_Q,
                               layer.attention.grad_W_K,
                               layer.attention.grad_W_V,
                               layer.attention.grad_W_O])
            # LayerNorm parameters
            self.params.extend([layer.layer_norm1.gamma,
                                layer.layer_norm1.beta,
                                layer.layer_norm2.gamma,
                                layer.layer_norm2.beta])
            self.grads.extend([layer.layer_norm1.grad_gamma,
                               layer.layer_norm1.grad_beta,
                               layer.layer_norm2.grad_gamma,
                               layer.layer_norm2.grad_beta])
            # FeedForward parameters
            self.params.extend([layer.feed_forward.fc1.weights,
                                layer.feed_forward.fc1.bias,
                                layer.feed_forward.fc2.weights,
                                layer.feed_forward.fc2.bias])
            self.grads.extend([layer.feed_forward.fc1.grad_weights,
                               layer.feed_forward.fc1.grad_bias,
                               layer.feed_forward.fc2.grad_weights,
                               layer.feed_forward.fc2.grad_bias])

        # Optimizer
        self.optimizer = AdamOptimizer(self.params, self.grads, lr=lr)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, seq_length]
        x = self.embedding.forward(x)  # [batch_size, seq_length, embed_size]
        x = self.pos_encoding.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        logits = self.output_proj.forward(x)  # [batch_size, seq_length, vocab_size]
        return logits

    def backward(self, logits: Tensor, labels: Tensor) -> None:
        # Cross-entropy loss gradient
        batch_size, seq_length, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)  # [(batch_size * seq_length), vocab_size]
        labels = labels.reshape(-1)  # [(batch_size * seq_length)]

        probs = F.softmax(logits, dim=-1)
        one_hot_labels = F.one_hot(labels, num_classes=vocab_size).float()
        loss_grad = (probs - one_hot_labels) / batch_size  # [(batch_size * seq_length), vocab_size]
        loss_grad = loss_grad.view(batch_size, seq_length, vocab_size)

        # Backward through output projection
        grad_x = self.output_proj.backward(loss_grad)
        # Backward through transformer layers
        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)
        # Backward through embedding
        self.embedding.backward(grad_x)

    def zero_grad(self):
        self.embedding.zero_grad()
        self.output_proj.zero_grad()
        for layer in self.layers:
            layer.zero_grad()

    def update_parameters(self):
        self.optimizer.step()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def generate(self,
                 initial_input: Tensor,
                 max_length: int,
                 temperature: float = 1.0,
                 frequency_penalty: float = 0.0,
                 stop_token: Optional[List[int]] = None,
                 greedy: bool = False) -> Tensor:
        # initial_input: [batch_size, seq_length]
        self.eval()
        batch_size = initial_input.shape[0]
        input_indices = initial_input.clone()
        generated = input_indices
        token_frequencies = torch.zeros(batch_size, self.vocab_size, device=device)

        stop_token_len = len(stop_token) if stop_token is not None else 0

        for _ in range(max_length - input_indices.shape[1]):
            logits = self.forward(input_indices)
            # Get the last token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Apply frequency penalty
            if frequency_penalty > 0.0:
                next_token_logits -= frequency_penalty * token_frequencies

            probs = F.softmax(next_token_logits, dim=-1)

            if greedy:
                next_token = torch.argmax(probs, dim=-1)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Update token frequencies
            token_frequencies[torch.arange(batch_size), next_token] += 1

            # Append next token
            input_indices = torch.cat([input_indices, next_token.unsqueeze(-1)], dim=-1)
            generated = input_indices

            # Check for stop token
            if stop_token is not None:
                if all([torch.equal(input_indices[i, -stop_token_len:], torch.tensor(stop_token, device=device))
                        for i in range(batch_size)]):
                    break

            # Truncate sequence if it exceeds max_seq_len
            if input_indices.shape[1] > self.max_seq_len:
                input_indices = input_indices[:, -self.max_seq_len:]

        return generated

    def save_model(self, path: str):
        """
        Saves the model weights to a file.
        """
        state_dict = {
            'embedding_weights': self.embedding.weights.cpu(),
            'output_W': self.output_proj.W.cpu(),
            'optimizer_state': {
                'm': [m.cpu() for m in self.optimizer.m],
                'v': [v.cpu() for v in self.optimizer.v],
                't': self.optimizer.t
            }
        }
        # Save layers
        for idx, layer in enumerate(self.layers):
            layer_state = {
                'W_Q': layer.attention.W_Q.cpu(),
                'W_K': layer.attention.W_K.cpu(),
                'W_V': layer.attention.W_V.cpu(),
                'W_O': layer.attention.W_O.cpu(),
                'layer_norm1_gamma': layer.layer_norm1.gamma.cpu(),
                'layer_norm1_beta': layer.layer_norm1.beta.cpu(),
                'layer_norm2_gamma': layer.layer_norm2.gamma.cpu(),
                'layer_norm2_beta': layer.layer_norm2.beta.cpu(),
                'fc1_weights': layer.feed_forward.fc1.weights.cpu(),
                'fc1_bias': layer.feed_forward.fc1.bias.cpu(),
                'fc2_weights': layer.feed_forward.fc2.weights.cpu(),
                'fc2_bias': layer.feed_forward.fc2.bias.cpu(),
            }
            state_dict[f'layer_{idx}'] = layer_state
        torch.save(state_dict, path)

    def load_model(self, path: str):
        """
        Loads the model weights from a file.
        """
        state_dict = torch.load(path, map_location=device)
        self.embedding.weights = state_dict['embedding_weights'].to(device)
        self.output_proj.W = state_dict['output_W'].to(device)

        # Optimizer state
        self.optimizer.m = [m.to(device) for m in state_dict['optimizer_state']['m']]
        self.optimizer.v = [v.to(device) for v in state_dict['optimizer_state']['v']]
        self.optimizer.t = state_dict['optimizer_state']['t']

        # Load layers
        for idx, layer in enumerate(self.layers):
            layer_state = state_dict[f'layer_{idx}']
            layer.attention.W_Q = layer_state['W_Q'].to(device)
            layer.attention.W_K = layer_state['W_K'].to(device)
            layer.attention.W_V = layer_state['W_V'].to(device)
            layer.attention.W_O = layer_state['W_O'].to(device)
            layer.layer_norm1.gamma = layer_state['layer_norm1_gamma'].to(device)
            layer.layer_norm1.beta = layer_state['layer_norm1_beta'].to(device)
            layer.layer_norm2.gamma = layer_state['layer_norm2_gamma'].to(device)
            layer.layer_norm2.beta = layer_state['layer_norm2_beta'].to(device)
            layer.feed_forward.fc1.weights = layer_state['fc1_weights'].to(device)
            layer.feed_forward.fc1.bias = layer_state['fc1_bias'].to(device)
            layer.feed_forward.fc2.weights = layer_state['fc2_weights'].to(device)
            layer.feed_forward.fc2.bias = layer_state['fc2_bias'].to(device)

    def train_model(self, data_loader, epochs: int, batch_size:int):
        self.train()
        loss_history = []
        for epoch in tqdm(range(epochs)):
            total_loss = 0.0
            for batch in data_loader:
                input_indices = batch.to(device)  # [batch_size, seq_length]
                # Prepare target labels
                labels = input_indices[:, 1:]  # [batch_size, seq_length-1]
                input_indices = input_indices[:, :-1]  # [batch_size, seq_length-1]

                # Forward pass
                logits = self.forward(input_indices)  # [batch_size, seq_length-1, vocab_size]

                # Calculate loss
                batch_size, seq_length, vocab_size = logits.shape
                logits_flat = logits.reshape(-1, vocab_size)
                labels_flat = labels.reshape(-1)
                loss = F.cross_entropy(logits_flat, labels_flat)
                total_loss += loss.item()

                # Backward pass
                self.backward(logits, labels)

                # Update parameters
                self.update_parameters()

                # Zero gradients
                self.zero_grad()
            avg_loss = total_loss / batch_size
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            loss_history.append(avg_loss)
        return loss