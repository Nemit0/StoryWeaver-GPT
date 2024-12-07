import torch
from torch import tensor, Tensor
from typing import List, Dict, Tuple
from tqdm import tqdm

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
        self.weights = torch.randn(input_size, output_size, dtype=torch.float64, requires_grad=False) * 0.01
        self.bias = torch.zeros(output_size, dtype=torch.float64, requires_grad=False)
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

class GeLUActivation:
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        c = torch.tensor(0.7978845608, dtype=x.dtype, device=x.device) # sqrt(2/pi) approximated
        a = torch.tensor(0.044715, dtype=x.dtype, device=x.device) # 2/(pi*sqrt(2)) approximated
        self.c = c
        self.a = a
        self.s = c * (x + a * x ** 3)
        self.tanh_s = torch.tanh(self.s)
        y = 0.5 * x * (1.0 + self.tanh_s)
        return y

    def backward(self, grad_output: Tensor) -> Tensor:
        x = self.input
        c = self.c
        a = self.a
        s = self.s
        tanh_s = self.tanh_s
        sech2_s = 1.0 - tanh_s ** 2
        ds_dx = c * (1.0 + 3.0 * a * x ** 2)
        dy_dx = 0.5 * (1.0 + tanh_s + x * sech2_s * ds_dx)
        grad_input = grad_output * dy_dx
        return grad_input


class Embedding:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights: Tensor = torch.randn(input_dim, output_dim, dtype=torch.float64, requires_grad=False) * 0.01
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
        self.pos_encoding = torch.zeros(max_seq_len, embed_size, dtype=torch.float64, requires_grad=False)

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
        self.W_Q = [torch.randn(embed_size, self.head_dim, dtype=torch.float64, requires_grad=False) * 0.01 for _ in range(heads)]
        self.W_K = [torch.randn(embed_size, self.head_dim, dtype=torch.float64, requires_grad=False) * 0.01 for _ in range(heads)]
        self.W_V = [torch.randn(embed_size, self.head_dim, dtype=torch.float64, requires_grad=False) * 0.01 for _ in range(heads)]
        self.W_O = [torch.randn(self.head_dim, embed_size, dtype=torch.float64, requires_grad=False) * 0.01 for _ in range(heads)]

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
            grad_V = self.attention_weights[i].transpose(0, 1) @ grad_attn_output  # (seq_len, head_dim)

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
        self.attention:MultiHeadAttention = MultiHeadAttention(embed_size, heads)
        self.layer_norm:LayerNorm = LayerNorm(embed_size)

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
        self.activation = GeLUActivation()
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
        self.W = torch.randn(embed_size, vocab_size, dtype=torch.float64, requires_grad=False) * 0.01
        self.grad_W = torch.zeros_like(self.W)

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        self.logits = x @ self.W
        return self.logits

    def backward(self, grad_output: Tensor) -> Tensor:
        self.grad_W += self.input.transpose(0, 1) @ grad_output
        grad_input = grad_output @ self.W.transpose(0, 1)
        return grad_input

class AdamOptimizer:
    def __init__(self, params: Dict[str, Tensor|Dict], grads: Dict[str, Tensor], lr: float = 1e-3, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        """
        Initializes the Adam optimizer.

        Args:
            params (Dict[str, Tensor]): Dictionary of parameters to optimize.
            grads (Dict[str, Tensor]): Dictionary of corresponding gradients.
            lr (float): Learning rate.
            betas (Tuple[float, float]): Coefficients used for computing running averages.
            eps (float): Term added to the denominator to improve numerical stability.
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.params = params
        self.grads = grads
        self.m = {key: torch.zeros_like(param) for key, param in self.params.items()}
        self.v = {key: torch.zeros_like(param) for key, param in self.params.items()}
        self.t = 0  # Time step

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        self.t += 1
        for key in self.params.keys():
            grad = self.grads.get(f"{key}_grad")
            if grad is None:
                continue

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            self.params[key] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        Resets all gradients to zero.
        """
        for grad in self.grads.values():
            grad.zero_()

class AdamOptimizer:
    def __init__(self, params: Dict[str, Tensor], grads: Dict[str, Tensor], lr: float = 1e-3, 
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        """
        Initializes the Adam optimizer.

        Args:
            params (Dict[str, Tensor]): Dictionary of parameters to optimize.
            grads (Dict[str, Tensor]): Dictionary of corresponding gradients.
            lr (float): Learning rate.
            betas (Tuple[float, float]): Coefficients used for computing running averages.
            eps (float): Term added to the denominator to improve numerical stability.
        """
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.params = params
        self.grads = grads
        self.m = {key: torch.zeros_like(param) for key, param in self.params.items()}
        self.v = {key: torch.zeros_like(param) for key, param in self.params.items()}
        self.t = 0  # Time step

    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        self.t += 1
        for key in self.params.keys():
            grad = self.grads.get(key)
            if grad is None:
                continue

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            self.params[key] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        Resets all gradients to zero.
        """
        for grad in self.grads.values():
            grad.zero_()

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
    def __init__(self, vocab_size: int, embed_size: int, max_seq_len: int, heads: int, ff_dim: int, num_blocks: int, lr: float = 1e-3):
        self.embed_size: int = embed_size
        self.max_seq_len: int = max_seq_len
        self.num_blocks: int = num_blocks
        self.token_embedding: Embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(max_seq_len, embed_size)
        self.transformer_blocks: List[TransformerEncoderBlock] = []
        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerEncoderBlock(embed_size, heads, ff_dim))
        self.output: OutputProjection = OutputProjection(embed_size, vocab_size)
        self.train_mode: bool = True

        # Parameter and gradient tracking
        self.param_and_grads: Dict = {
            "embedding_weight": self.token_embedding.weights,
            "embedding_weight_grad": self.token_embedding.grad_weights,
            "transformer_block": [
                {
                    "attention": {
                        "W_Q": block.attention.attention.W_Q,
                        "W_K": block.attention.attention.W_K,
                        "W_V": block.attention.attention.W_V,
                        "W_O": block.attention.attention.W_O,
                        "W_Q_grad": block.attention.attention.grad_W_Q,
                        "W_K_grad": block.attention.attention.grad_W_K,
                        "W_V_grad": block.attention.attention.grad_W_V,
                        "W_O_grad": block.attention.attention.grad_W_O,
                    },
                    "layernorm_1": {
                        "gamma": block.layer_norm_1.gamma,
                        "beta": block.layer_norm_1.beta,
                        "gamma_grad": block.layer_norm_1.grad_gamma,
                        "beta_grad": block.layer_norm_1.grad_beta,
                    },
                    "FeedForward": {
                        "fc1_weights": block.feed_forward.fc1.weights,
                        "fc1_bias": block.feed_forward.fc1.bias,
                        "fc1_weights_grad": block.feed_forward.fc1.grad_weights,
                        "fc1_bias_grad": block.feed_forward.fc1.grad_bias,
                        "fc2_weights": block.feed_forward.fc2.weights,
                        "fc2_bias": block.feed_forward.fc2.bias,
                        "fc2_weights_grad": block.feed_forward.fc2.grad_weights,
                        "fc2_bias_grad": block.feed_forward.fc2.grad_bias,
                    },
                    "layernorm_2": {
                        "gamma": block.layer_norm_2.gamma,
                        "beta": block.layer_norm_2.beta,
                        "gamma_grad": block.layer_norm_2.grad_gamma,
                        "beta_grad": block.layer_norm_2.grad_beta,
                    }
                } for block in self.transformer_blocks
            ],
            "output_weight": self.output.W,
            "output_weight_grad": self.output.grad_W,
        }

        # Prepare parameters and grads
        param_dict = {
            "embedding_weight": self.token_embedding.weights,
            "embedding_weight_grad": self.token_embedding.grad_weights,
            "output_weight": self.output.W,
            "output_weight_grad": self.output.grad_W
        }

        # Add transformer block parameters
        for i, block in enumerate(self.transformer_blocks):
            # Attention Parameters
            # Flatten Multihead attention parameters
            for head_idx in range(block.attention.attention.heads):
                param_dict[f"transformer_block_{i}_attention_W_Q_{head_idx}"] = block.attention.attention.W_Q[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_Q_{head_idx}_grad"] = block.attention.attention.grad_W_Q[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_K_{head_idx}"] = block.attention.attention.W_K[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_K_{head_idx}_grad"] = block.attention.attention.grad_W_K[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_V_{head_idx}"] = block.attention.attention.W_V[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_V_{head_idx}_grad"] = block.attention.attention.grad_W_V[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_O_{head_idx}"] = block.attention.attention.W_O[head_idx]
                param_dict[f"transformer_block_{i}_attention_W_O_{head_idx}_grad"] = block.attention.attention.grad_W_O[head_idx]

            # Layer Norm in AttentionBlock
            param_dict[f"transformer_block_{i}_attention_layernorm_gamma"] = block.attention.layer_norm.gamma
            param_dict[f"transformer_block_{i}_attention_layernorm_gamma_grad"] = block.attention.layer_norm.grad_gamma
            param_dict[f"transformer_block_{i}_attention_layernorm_beta"] = block.attention.layer_norm.beta
            param_dict[f"transformer_block_{i}_attention_layernorm_beta_grad"] = block.attention.layer_norm.grad_beta

            # Layer Norm 1
            param_dict[f"transformer_block_{i}_layernorm_1_gamma"] = block.layer_norm_1.gamma
            param_dict[f"transformer_block_{i}_layernorm_1_gamma_grad"] = block.layer_norm_1.grad_gamma
            param_dict[f"transformer_block_{i}_layernorm_1_beta"] = block.layer_norm_1.beta
            param_dict[f"transformer_block_{i}_layernorm_1_beta_grad"] = block.layer_norm_1.grad_beta

            # Layer Norm 2
            param_dict[f"transformer_block_{i}_layernorm_2_gamma"] = block.layer_norm_2.gamma
            param_dict[f"transformer_block_{i}_layernorm_2_gamma_grad"] = block.layer_norm_2.grad_gamma
            param_dict[f"transformer_block_{i}_layernorm_2_beta"] = block.layer_norm_2.beta
            param_dict[f"transformer_block_{i}_layernorm_2_beta_grad"] = block.layer_norm_2.grad_beta

            # FeedForward
            param_dict[f"transformer_block_{i}_fc1_weights"] = block.feed_forward.fc1.weights
            param_dict[f"transformer_block_{i}_fc1_weights_grad"] = block.feed_forward.fc1.grad_weights
            param_dict[f"transformer_block_{i}_fc1_bias"] = block.feed_forward.fc1.bias
            param_dict[f"transformer_block_{i}_fc1_bias_grad"] = block.feed_forward.fc1.grad_bias

            param_dict[f"transformer_block_{i}_fc2_weights"] = block.feed_forward.fc2.weights
            param_dict[f"transformer_block_{i}_fc2_weights_grad"] = block.feed_forward.fc2.grad_weights
            param_dict[f"transformer_block_{i}_fc2_bias"] = block.feed_forward.fc2.bias
            param_dict[f"transformer_block_{i}_fc2_bias_grad"] = block.feed_forward.fc2.grad_bias

        # Separate params and grads for optimizer
        params = {}
        grads = {}
        for k, v in param_dict.items():
            if "_grad" in k:
                grads[k.replace("_grad", "")] = v
            else:
                params[k] = v

        self.optimizer = AdamOptimizer(params=params, grads=grads, lr=lr)

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
        batch_size, vocab_size = probs.shape
        labels_one_hot = torch.zeros_like(probs)
        labels_one_hot[range(batch_size), labels] = 1

        grad_logits = (probs - labels_one_hot) / batch_size  # Gradient w.r.t. logits

        # Backpropagate through output projection
        grad_output = self.output.backward(grad_logits)

        # Backpropagate through Transformer blocks
        for block in reversed(self.transformer_blocks):
            grad_output = block.backward(grad_output)

        # Backpropagate through token embedding
        self.token_embedding.backward(grad_output)

    def update_parameters(self):
        """
        Updates parameters using the Adam optimizer.
        """
        self.optimizer.step()

    def zero_grad(self):
        """
        Resets all gradients using the Adam optimizer.
        """
        self.optimizer.zero_grad()

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
    
    def check_gradients(self):
        nan_in_gradients = False
        # Check token embedding gradients
        if torch.any(torch.isnan(self.token_embedding.grad_weights)):
            print("NaN detected in token_embedding.grad_weights")
            nan_in_gradients = True
        # TODO: Check other params
        return nan_in_gradients

    def check_parameters(self):
        nan_in_params = False
        # Check token embedding weights
        if torch.any(torch.isnan(self.token_embedding.weights)):
            print("NaN detected in token_embedding.weights")
            nan_in_params = True
        # TODO: Check other params
        return nan_in_params
    
    def clip_gradients(self, max_norm):
        # Clip token embedding gradients
        torch.nn.utils.clip_grad_norm_([self.token_embedding.grad_weights], max_norm)
        # Clip output projection gradients
        torch.nn.utils.clip_grad_norm_([self.output.grad_W], max_norm)
        # Clip gradients in transformer blocks
        for block in self.transformer_blocks:
            attention = block.attention.attention
            for i in range(attention.heads):
                torch.nn.utils.clip_grad_norm_([attention.grad_W_Q[i]], max_norm)
                torch.nn.utils.clip_grad_norm_([attention.grad_W_K[i]], max_norm)
                torch.nn.utils.clip_grad_norm_([attention.grad_W_V[i]], max_norm)
                torch.nn.utils.clip_grad_norm_([attention.grad_W_O[i]], max_norm)
            # Clip layer norm gradients
            torch.nn.utils.clip_grad_norm_([block.attention.layer_norm.grad_gamma, block.attention.layer_norm.grad_beta], max_norm)
            torch.nn.utils.clip_grad_norm_([block.layer_norm_1.grad_gamma, block.layer_norm_1.grad_beta], max_norm)
            torch.nn.utils.clip_grad_norm_([block.layer_norm_2.grad_gamma, block.layer_norm_2.grad_beta], max_norm)
            # Clip feed-forward gradients
            torch.nn.utils.clip_grad_norm_([block.feed_forward.fc1.grad_weights, block.feed_forward.fc1.grad_bias], max_norm)
            torch.nn.utils.clip_grad_norm_([block.feed_forward.fc2.grad_weights, block.feed_forward.fc2.grad_bias], max_norm)

    def train_model(self, data: List[Tensor], epochs: int, learning_rate: float) -> List[float]:
        loss_history = []
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            for input_indices in data:
                labels = input_indices[1:]  # Shifted input indices (next tokens)
                input_indices = input_indices[:-1]  # Input tokens

                # Forward pass
                probs = self.forward(input_indices)
                # Add epsilon to prevent log(0)
                eps = 1e-10
                probs_correct = probs[range(len(labels)), labels] + eps
                loss = -torch.log(probs_correct).mean()
                epoch_loss += loss.item()

                # Backward pass
                self.backward(probs, labels)

                # Gradient Clipping
                self.clip_gradients(max_norm=1.0)

                # Update parameters using Adam
                self.update_parameters()

                # Check for NaNs in parameters
                if self.check_parameters():
                    print("NaN detected in parameters. Stopping training.")
                    return loss_history

                # Zero gradients
                self.zero_grad()

            avg_loss = epoch_loss / len(data)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # Early stopping if loss is NaN
            if torch.isnan(torch.tensor(avg_loss)):
                print("Loss became NaN. Stopping training.")
                break

        return loss_history
    
    def eval_mode(self):
        """
        Sets the model to evaluation mode.
        """
        self.train_mode = False   

    def generate_sequence(self, initial_input, max_length):
        self.eval_mode()
        input_indices = initial_input.clone()

        for _ in range(max_length - len(initial_input)):
            # Forward pass
            probs = self.forward(input_indices)
            # Get the last token's probability distribution
            next_token_probs = probs[-1]
            # Sample the next token (you can also use argmax for deterministic results)
            next_token = torch.argmax(next_token_probs)
            # Append the next token to the input sequence
            input_indices = torch.cat((input_indices, next_token.unsqueeze(0)), dim=0)
            # If input_indices length exceeds max_seq_len, truncate it
            if len(input_indices) > self.max_seq_len:
                input_indices = input_indices[-self.max_seq_len:]
        return input_indices
    
def generate_sequence(model, initial_input, max_length):
    model.eval_mode = True  # Ensure the model is in evaluation mode
    input_indices = initial_input.clone()
    
    for _ in range(max_length - len(initial_input)):
        # Forward pass
        probs = model.forward(input_indices)
        # Get the last token's probability distribution
        next_token_probs = probs[-1]
        # Sample the next token (you can also use argmax for deterministic results)
        next_token = torch.argmax(next_token_probs)
        # Append the next token to the input sequence
        input_indices = torch.cat((input_indices, next_token.unsqueeze(0)), dim=0)
        # If input_indices length exceeds max_seq_len, truncate it
        if len(input_indices) > model.max_seq_len:
            input_indices = input_indices[-model.max_seq_len:]
    return input_indices