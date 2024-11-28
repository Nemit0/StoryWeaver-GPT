import json
import re
import os
import torch
from torch import tensor, Tensor
from typing import List, Dict, Tuple
from multiprocessing import Pool
from tqdm import tqdm
from nn_objects import Layer, Linear, ReLU, CrossEntropyLoss

### CUDA SETUP ###

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_device(device)
    print(f"Using {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    torch.set_default_device(device)
    print("Using CPU")

### TOKENIZER ###

class BytePairTokenizer:
    def __init__(self, data_path:str=None) -> None:
        """
        BytePairTokenizer object
        """
        if data_path:
            self.load_model(data_path)
            return
        
        self.special_tokens:Dict[str, int] = {
            '<BOT>': 0,  # Beginning of Text
            '<EOT>': 1,   # End of Text
            '</w>': 2     # end of word
        }
        self.inv_special_tokens:Dict[int, str] = {i: t for t, i in self.special_tokens.items()}

        self.token_map: Dict[str, int] = self.special_tokens.copy()
        self.inv_map: Dict[int, str] = self.inv_special_tokens.copy()
        self.bpe_codes: Dict[Tuple[str, str], int] = {}
    
    def train(self, corpus: List[str], num_merges: int, verbose:bool = False) -> None:
        """
        Train the Byte Pair Tokenizer to process sentences.
        """
        # Build the vocabulary: map token sequences to their frequencies
        vocab = {}
        if verbose:
            print("Building vocabulary...")
        for sentence in tqdm(corpus):
            # Split sentence into words with leading whitespace preserved
            words = re.findall(r'\s*\S+|\s+', sentence)
            for word in words:
                # Skip special tokens
                if word in self.special_tokens.keys():
                    continue
                chars = list(word) + ['</w>']
                word_tuple = tuple(chars)
                vocab[word_tuple] = vocab.get(word_tuple, 0) + 1
        
        if verbose:
            print("Vocabulary built.\nTraining BPE...")
        token_id = len(self.token_map)  # Starting token ID
        symbols = set()
        for word_tuple in vocab.keys():
            symbols.update(word_tuple)
        for symbol in symbols:
            if symbol not in self.token_map:
                self.token_map[symbol] = token_id
                token_id += 1
        self.inv_map = {i: t for t, i in self.token_map.items()}
        
        if verbose:
            print("Token map built.\nMerging tokens...")
        # Perform BPE merges
        for i in tqdm(range(num_merges)):
            pairs = self._get_pair_counts(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.bpe_codes[best_pair] = i # Record the BPE merge rule
            new_symbol = ''.join(best_pair)
            if new_symbol not in self.token_map:
                self.token_map[new_symbol] = token_id
                token_id += 1
                self.inv_map[self.token_map[new_symbol]] = new_symbol
    
    def _get_pair_counts(self, vocab: Dict[Tuple[str], int]) -> Dict[Tuple[str, str], int]:
        """
        Get counts of symbol pairs in the vocabulary
        """
        pairs = {}
        for word, freq in vocab.items():
            symbols = word
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    def _merge_vocab_single(self, pair: Tuple[str, str], vocab: Dict[Tuple[str], int]) -> Dict[Tuple[str], int]:
        """
        Merge all occurrences of the given pair in the vocabulary
        """
        new_vocab = {}
        bigram = ''.join(pair)
        for word, freq in vocab.items():
            w = []
            i = 0
            while i < len(word):
                # Merge the pair if found
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    w.append(bigram)
                    i += 2
                else:
                    w.append(word[i])
                    i += 1
            new_vocab[tuple(w)] = freq
        return new_vocab

    @staticmethod
    def _process_word(args):
        pair, word_freq = args
        word, freq = word_freq
        bigram = ''.join(pair)
        w = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                w.append(bigram)
                i += 2
            else:
                w.append(word[i])
                i += 1
        return tuple(w), freq
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str], int]) -> Dict[Tuple[str], int]:
        """
        Parallel merge of all occurrences of the given pair in the vocabulary using multiprocessing.
        """
        with Pool() as pool:
            results = pool.map(self._process_word, [(pair, word_freq) for word_freq in vocab.items()])

        new_vocab = {word: freq for word, freq in results}
        return new_vocab
    
    def _get_pairs(self, word: List[str]) -> set:
        """
        Return a set of symbol pairs in a word
        """
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
    
    def _apply_bpe(self, word: List[str]) -> List[str]:
        """
        Apply BPE to a list of symbols (a word)
        """
        word = word.copy()
        pairs = self._get_pairs(word)
        while True:
            if not pairs:
                break
            # Find the highest priority pair to merge
            min_pair = None
            min_rank = float('inf')
            for pair in pairs:
                if pair in self.bpe_codes:
                    rank = self.bpe_codes[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair
            if min_pair is None:
                break
            # Merge the best pair
            new_symbol = ''.join(min_pair)
            i = 0
            while i < len(word) - 1:
                if word[i] == min_pair[0] and word[i + 1] == min_pair[1]:
                    word[i:i + 2] = [new_symbol]
                    i = max(i - 1, 0)  # Restart from the previous position after a merge
                else:
                    i += 1
            pairs = self._get_pairs(word)
        return word
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into BPE tokens with leading whitespace preserved
        """
        tokens = []
        words = re.findall(r'\s*\S+|\s+', text)
        for word in words:
            chars = list(word) + ['</w>']
            bpe_word = self._apply_bpe(chars)
            tokens.extend(bpe_word)
        return tokens
    
    def encode(self, data: str) -> List[int]:
        """
        Encode text data into a list of token IDs
        """
        str_list = self.split_text(data)
        token_list = [self.token_map[tok] for tok in str_list]
        return token_list
    
    def decode(self, data: List[int]) -> str:
        """
        Decode a list of token IDs back into text
        """
        tokens = [self.inv_map[i] for i in data]
        text = ''
        for token in tokens:
            if token != '</w>':
                text += token.replace('</w>', '')
        return text

    def save_model(self, target_path:str) -> None:
        """
        Save the model to a file as json file
        the json will look like
        {
            token_map : {...},
            bpe_codes : {...}
        }
        The special tokens are not necessary for simple encoding/decoding
        hence it is omitted from the model
        """
        with open(target_path, 'w', encoding="UTF-8") as f:
            json.dump({
                'token_map': self.token_map,
                'bpe_codes': {json.dumps(list(k)): v for k, v in self.bpe_codes.items()}
            }, f,
             indent=4,
              ensure_ascii=False)
    
    def load_model(self, model_path:str, encoding="UTF-8") -> None:
        """
        Load the model from a json file
        JSON doesn't allow tuple object as key
        hence the tuple keys are converted to string before saving
        and converted back to tuple when loading
        """
        with open(model_path, 'r') as f:
            model = json.load(f)
        self.token_map = model['token_map']
        self.inv_map = {i: t for t, i in self.token_map.items()}
        self.bpe_codes = {tuple(json.loads(k)): v for k, v in model['bpe_codes'].items()}

def load_tokenizer(path:str = None) -> BytePairTokenizer:
    """
    Load the BytePairTokenizer model from the model folder
    """
    if path is None:
        model_path:str = os.path.join(os.getcwd(), 'model', 'tokenizer.json')
    else:
        model_path:str = path
    tokenizer = BytePairTokenizer(model_path)
    # tokenizer.load_model(model_path)
    return tokenizer

### NEURAL NETWORK OBJECTS ###

class SimpleLinear:
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Args:
            input_size (int): 입력 피처의 크기
            output_size (int): 출력 피처의 크기
        """
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.weights: Tensor = torch.rand(input_size, output_size) # 가중치 랜덤 초기화

    def forward(self, inputs: Tensor) -> Tensor:
        """
        입력에 가중치를 단순 행렬곱하여 출력

        Args:
            inputs (Tensor): 입력 텐서 [batch_size, input_size]

        Returns:
            Tensor: 출력 텐서 [batch_size, output_size]
        """
        self.inputs: Tensor = inputs
        self.output: Tensor = torch.mm(inputs, self.weights) # 단순 행렬곱
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        손실 함수 그래디언트 이전 층으로 전달 및 가중치 그래디언트 계산

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트 [batch_size, output_size]

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트 [batch_size, input_size]
        """

        grad_input: Tensor = torch.mm(grad_output, self.weights.t()) # 단순 행렬곱
        self.grad_weights: Tensor = torch.mm(self.inputs.t(), grad_output)
        return grad_input

class Embedding:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Custom Embedding layer initialization
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize embedding matrix as learnable parameters
        self.weights: Tensor = torch.randn(input_dim, output_dim) * 0.01
        self.grad_weights: Tensor = torch.zeros_like(self.weights)

    def forward(self, input_indices: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            input_indices (Tensor): Integer index tensor (e.g., [batch_size, sequence_length])

        Returns:
            Tensor: Embedded vector tensor (e.g., [batch_size, sequence_length, output_dim])
        """
        self.input_indices = input_indices
        # Select embedding vectors using indices
        self.output = self.weights[input_indices]
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트 (예: [batch_size, sequence_length, output_dim])

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트 (임베딩 레이어의 경우 없음)
        """
        # grad_output의 형태: [batch_size, sequence_length, output_dim]
        # 이를 [batch_size * sequence_length, output_dim]로 평탄화
        grad_flat = grad_output.view(-1, self.output_dim)
        # input_indices를 평탄화하여 [batch_size * sequence_length] 형태로 
        input_flat = self.input_indices.view(-1)
        
        # 그래디언트를 초기화
        self.grad_weights.zero_()
        # 그래디언트 누적
        self.grad_weights.index_add_(0, input_flat, grad_flat)
        
        return None

    def __str__(self) -> str:
        return "CustomEmbedding"

class PositionalEncoding:
    def __init__(self, max_seq_len: int, embed_size: int):
        """
        Positional Encoding initialization
        """
        self.embed_size = embed_size
        self.pos_encoding = torch.zeros(max_seq_len, embed_size)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-torch.log(torch.tensor(10000.0)) / embed_size))
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x (Tensor): Embedded input tensor [seq_length, embed_size]

        Returns:
            Tensor: Tensor with positional encoding added [seq_length, embed_size]
        """
        seq_length, embed_size = x.shape

        # Ensure positional encoding matches input size
        pos_encoding = self.pos_encoding[:seq_length, :]  # Slice for the current sequence length

        return x + pos_encoding.to(x.device)
   
class MultiHeadAttention:
    def __init__(self, embed_size: int, heads: int):
        """
        Multi-Head Attention 초기화

        Args:
            embed_size (int): 임베딩 차원
            heads (int): 어텐션 헤드 수
        """
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        if embed_size % heads != 0:
            raise ValueError(f"Embedding dimension ({embed_size}) must be divisible by the number of heads ({heads})")

        # Initialize separate weights for each head
        self.W_Q = [torch.randn(embed_size, self.head_dim) * (self.head_dim ** -0.5) for _ in range(heads)]
        self.W_K = [torch.randn(embed_size, self.head_dim) * (self.head_dim ** -0.5) for _ in range(heads)]
        self.W_V = [torch.randn(embed_size, self.head_dim) * (self.head_dim ** -0.5) for _ in range(heads)]
        self.W_O = [torch.randn(self.head_dim, embed_size) * (self.head_dim ** -0.5) for _ in range(heads)]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for multi-head attention with masking.

        Args:
            x (Tensor): Input tensor of shape [seq_length, embed_size].

        Returns:
            Tensor: Output tensor of shape [seq_length, embed_size].
        """
        seq_length, embed_size = x.size()
        assert embed_size == self.embed_size, "Input embedding size must match initialized size."

        # Compute Q, K, V for each head
        Q_heads, K_heads, V_heads = [], [], []
        for i in range(self.heads):
            Q_heads.append(torch.matmul(x, self.W_Q[i]))  # [seq_length, head_dim]
            K_heads.append(torch.matmul(x, self.W_K[i]))  # [seq_length, head_dim]
            V_heads.append(torch.matmul(x, self.W_V[i]))  # [seq_length, head_dim]

        # Create a mask for the lower triangular part
        mask = torch.triu(torch.full((seq_length, seq_length), float('-inf')), diagonal=1)  # [seq_length, seq_length]

        # Compute attention for each head
        head_outputs = []
        for i in range(self.heads):
            # Scaled dot-product attention with masking
            scores = torch.matmul(Q_heads[i], K_heads[i].transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # [seq_length, seq_length]
            scores = scores + mask  # Apply mask
            attention_weights = torch.softmax(scores, dim=-1)  # [seq_length, seq_length]
            attention_out = torch.matmul(attention_weights, V_heads[i])  # [seq_length, head_dim]
            head_outputs.append(torch.matmul(attention_out, self.W_O[i]))  # [seq_length, embed_size]

        # Combine outputs from all heads
        # print(torch.sum(torch.stack(head_outputs, dim=0), dim=0).shape)
        out = torch.sum(torch.stack(head_outputs, dim=0), dim=0)  # [seq_length, embed_size]
        return out

class AttentionBlock:
    def __init__(self, embed_size: int, heads: int):
        """
        Attention Block 초기화

        Args:
            embed_size (int): 임베딩 차원
            heads (int): Multi-Head Attention의 헤드 수
        """
        self.attention = MultiHeadAttention(embed_size, heads)

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [seq_length, embed_size]

        Returns:
            Tensor: Attention Block 출력 [seq_length, embed_size]
        """
        # Multi-Head Attention 수행
        attention_out = self.attention.forward(x)
        return attention_out

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트
        """
        raise NotImplementedError
        grad = self.layer_norm.backward(grad_output)
        return grad
        
class LayerNorm:
    def __init__(self, embed_size: int, eps: float = 1e-5):
        """
        레이어 정규화 초기화

        Args:
            embed_size (int): 임베딩 차원
            eps (float, optional): 안정성을 위한 작은 값. Defaults to 1e-5.
        """
        self.gamma = torch.ones(embed_size, requires_grad=False)
        self.beta = torch.zeros(embed_size, requires_grad=False)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [batch_size, seq_length, embed_size]

        Returns:
            Tensor: 정규화된 텐서
        """
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = x.std(dim=-1, keepdim=True)
        self.normalized = (x - self.mean) / (self.std + self.eps)
        return self.gamma * self.normalized + self.beta

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트
        """
        # 단순화를 위해 역전파는 gamma에 대한 기울기만 처리
        return grad_output * self.gamma
    
class FeedForward:
    def __init__(self, embed_size: int, forward_expansion: int):
        """
        피드포워드 네트워크 초기화

        Args:
            embed_size (int): 임베딩 차원
            forward_expansion (int): 피드포워드 네트워크의 확장 비율
            activation (Activation): 활성화 함수
        """
        self.fc1 = Layer(embed_size, embed_size * forward_expansion, ReLU()) 
        self.fc2 = Layer(embed_size * forward_expansion, embed_size, Linear()) 

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [batch_size, seq_length, embed_size]

        Returns:
            Tensor: 피드포워드 네트워크 출력
        """ 
        out = self.fc1.forward(x)
        out = self.fc2.forward(out)
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        역전파 과정

        Args:
            grad_output (Tensor): 상위 레이어로부터 전달된 그래디언트

        Returns:
            Tensor: 하위 레이어로 전달할 그래디언트
        """
        grad = self.fc2.backward(grad_output)
        grad = self.fc1.backward(grad)
        return grad

class OutputProjection:
    def __init__(self, embed_size: int, vocab_size: int):
        """
        Output Projection Layer 초기화

        Args:
            embed_size (int): 임베딩 차원
            vocab_size (int): 어휘 크기
        """
        self.W = torch.randn(embed_size, vocab_size) * 0.01  # 가중치 초기화

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [seq_length, embed_size]

        Returns:
            Tensor: 확률 분포를 위한 출력 [seq_length, vocab_size]
        """
        return torch.matmul(x, self.W)

class TransformerEncoderBlock:
    def __init__(self, embed_size: int, heads: int, ff_expand_dim: int, vocab_size: int):
        """
        Transformer Encoder Block 초기화

        Args:
            embed_size (int): 임베딩 차원
            heads (int): Attention 헤드 수
            ff_dim (int): Feed Forward 내부 차원
        """
        self.attention = AttentionBlock(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, ff_expand_dim)
        self.layer_norm_1 = LayerNorm(embed_size)
        self.layer_norm_2 = LayerNorm(embed_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        순전파 과정

        Args:
            x (Tensor): 입력 텐서 [seq_length, embed_size]

        Returns:
            Tensor: 출력 텐서 [seq_length, embed_size]
        """
        # Attention Block + Residual Connection
        attention_out = self.attention.forward(x)
        x = self.layer_norm_1.forward(x + attention_out)

        # Feed Forward + Residual Connection
        feed_forward_out = self.feed_forward.forward(x)
        out = self.layer_norm_2.forward(x + feed_forward_out)
        return out

class GPT:
    def __init__(self, vocab_size: int, embed_size: int, max_seq_len: int, heads: int, ff_dim: int, num_blocks: int):
        """
        Initialize a GPT model
        
        Args:
            vocab_size (int): Vocabulary size
            embed_size (int): Embedding dimension
            max_seq_len (int): Maximum sequence length
            heads (int): Number of attention heads
            ff_dim (int): Feed forward dimension
            num_blocks (int): Number of transformer layers
        """
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks
        self.token_embedding = Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(max_seq_len, embed_size)
        self.transformer_blocks = []
        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerEncoderBlock(embed_size, heads, ff_dim, vocab_size))
        self.output = OutputProjection(embed_size, vocab_size)
        self.train:bool = True
        self.loss = CrossEntropyLoss()
        self.param_count = embed_size * vocab_size + embed_size * max_seq_len + embed_size * embed_size * 4 * num_blocks + embed_size * vocab_size
        
    def forward(self, x: Tensor, temperature:float = 1.0, pos: PositionalEncoding=None) -> Tensor:
        """
        Forward pass through the GPT model
        
        Args:
            x (Tensor): Input tensor [seq_length]
        
        Returns:
            Tensor: Output tensor [seq_length, vocab_size]
        """
        x = self.token_embedding.forward(x)
        if self.train:
            x = self.positional_encoding.forward(x)
        else:
            if pos is None:
                raise ValueError("Positional encoding must be provided for inference")
            x = pos.forward(x)
        for block in self.transformer_blocks:
            x = block.forward(x)
        
        x = self.output.forward(x)

        x = torch.softmax(x / temperature, dim=-1)
        
        if self.train:
            return x
        else:
            # returns a predicted token with temperature applied
            prob = torch.distributions.Categorical(probs=x[-1])
            return prob.sample()
        
    def generate(self, x: Tensor, temperature: float = 1.0, max_tokens: int = 100) -> Tensor:
        """
        Generate a sequence of tokens
        
        Args:
            x (Tensor): Seed tensor [seq_length]
            temperature (float): Temperature for sampling
            max_tokens (int): Maximum number of tokens to generate
        
        Returns:
            Tensor: Generated token sequence [seq_length + max_tokens]
        """
        if self.train:
            raise ValueError("Model must be in eval mode for generation")

        for _ in range(max_tokens):
            seq_length = x.shape[0]
            pos = PositionalEncoding(seq_length, self.embed_size)  # Update positional encoding for the current sequence length
            next_token = self.forward(x, temperature, pos)  # Generate the next token
            x = torch.cat([x, next_token])  # Append the generated token to the sequence

        return x.tolist()
    
    def backward(self, logit:Tensor, lables:Tensor, *args, **kwargs) -> None:
        """
        Backward pass through the model
        """
        raise NotImplementedError
        # Encode the lables into one-hot encoding
        one_hot = torch.zeros_like(logit)
        one_hot.scatter_(1, lables.unsqueeze(1), 1)

        # compute cross-entropy loss
        

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
    embedding_dim = 1024
    max_seq_len = 1024
    heads = 16
    ff_expand_dim = 4
    input_indices = torch.tensor(encoded)

    # Create GPT model
    Gpt_Object = GPT(vocab_size, embedding_dim, max_seq_len, heads, ff_expand_dim, num_blocks=8)
    sample_logit = Gpt_Object.forward(input_indices)
    with open("./sample_logit.json", "w", encoding='utf-8') as f:
        json.dump(sample_logit.tolist(), f, indent=4, ensure_ascii=False)
    
    Gpt_Object.train = False
    pos = PositionalEncoding(max_seq_len, embedding_dim)
    output = Gpt_Object.forward(input_indices, pos=pos)
    print(output.item())
    print(f"GPT output shape: {output.shape}")
    print(f"Decoded token: {tokenizer.decode([output.item()])}")

    # Generate a sequence
    generated = Gpt_Object.generate(input_indices, temperature=1, max_tokens=100)
    print(f"Generated sequence: {tokenizer.decode(generated)}")

if __name__ == "__main__":
    main()