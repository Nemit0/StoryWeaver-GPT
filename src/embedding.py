import torch
from torch import Tensor
import math
import random

class Embedding:
    def __init__(self, vocab_size: int, embed_size: int) -> None:
        """
        Custom Embedding layer initialization.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of the embedding vectors.
        """
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # Initialize embedding matrix with small random values
        self.weights: Tensor = torch.randn(vocab_size, embed_size) * 0.01
        self.grad_weights: Tensor = torch.zeros_like(self.weights)
        # Placeholder for last input indices during forward pass
        self.input_indices = None

    def forward(self, input_indices: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_indices (Tensor): Tensor of word indices (e.g., context words indices).

        Returns:
            Tensor: Embedded vectors corresponding to input indices.
        """
        self.input_indices = input_indices
        # Select embeddings for the given indices
        self.output = self.weights[input_indices]
        return self.output

    def backward(self, grad_output: Tensor) -> None:
        """
        Backward pass.

        Args:
            grad_output (Tensor): Gradient from the next layer.

        Returns:
            None
        """
        # Flatten grad_output and input_indices for easy indexing
        grad_output_flat = grad_output.view(-1, self.embed_size)
        input_indices_flat = self.input_indices.view(-1)
        # Reset gradients
        self.grad_weights.zero_()
        # Accumulate gradients for embeddings
        for i in range(len(input_indices_flat)):
            idx = input_indices_flat[i]
            self.grad_weights[idx] += grad_output_flat[i]

    def update_weights(self, learning_rate: float) -> None:
        """
        Update the embedding weights using the accumulated gradients.

        Args:
            learning_rate (float): Learning rate for updating weights.
        """
        self.weights -= learning_rate * self.grad_weights

    def __str__(self) -> str:
        return "CustomEmbedding"

class CBOWModel:
    def __init__(self, vocab_size: int, embed_size: int) -> None:
        """
        Initialize CBOW model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_size (int): Dimension of the embedding vectors.
        """
        self.embedding = Embedding(vocab_size, embed_size)
        # Output weights and biases
        self.output_weights: Tensor = torch.randn(embed_size, vocab_size) * 0.01
        self.output_bias: Tensor = torch.zeros(vocab_size)
        self.grad_output_weights: Tensor = torch.zeros_like(self.output_weights)
        self.grad_output_bias: Tensor = torch.zeros_like(self.output_bias)
        # Placeholder for last hidden state
        self.hidden = None

    def forward(self, context_indices: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            context_indices (Tensor): Tensor of context word indices.

        Returns:
            Tensor: Output scores (before softmax) for each word in the vocabulary.
        """
        # Get embeddings for context words
        embeds = self.embedding.forward(context_indices)
        # Sum (or average) embeddings to get context vector
        self.hidden = embeds.mean(dim=0)
        # Compute scores
        scores = torch.matmul(self.hidden, self.output_weights) + self.output_bias
        return scores

    def backward(self, target_index: int, scores: Tensor, learning_rate: float) -> None:
        """
        Backward pass.

        Args:
            target_index (int): Index of the target word.
            scores (Tensor): Output scores from the forward pass.
            learning_rate (float): Learning rate for updates.
        """
        # Compute softmax probabilities
        probs = self.softmax(scores)
        # Compute loss derivative w.r.t. scores
        d_scores = probs
        d_scores[target_index] -= 1  # Subtract 1 from the true class
        # Compute gradients for output weights and biases
        d_output_weights = torch.outer(self.hidden, d_scores)
        d_output_bias = d_scores
        # Compute gradients w.r.t. hidden layer
        d_hidden = torch.matmul(self.output_weights, d_scores)
        # Backpropagate to embeddings
        grad_embeds = d_hidden.unsqueeze(0).expand_as(self.embedding.output) / self.embedding.output.shape[0]
        self.embedding.backward(grad_embeds)
        # Update output weights and biases
        self.grad_output_weights = d_output_weights
        self.grad_output_bias = d_output_bias
        self.update_weights(learning_rate)

    def update_weights(self, learning_rate: float) -> None:
        """
        Update weights using the accumulated gradients.

        Args:
            learning_rate (float): Learning rate for updating weights.
        """
        # Update embedding weights
        self.embedding.update_weights(learning_rate)
        # Update output weights and biases
        self.output_weights -= learning_rate * self.grad_output_weights
        self.output_bias -= learning_rate * self.grad_output_bias

    def softmax(self, x: Tensor) -> Tensor:
        """
        Compute softmax function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Softmax probabilities.
        """
        exps = torch.exp(x - torch.max(x))  # For numerical stability
        return exps / torch.sum(exps)

    def train(self, training_data: list, epochs: int, learning_rate: float) -> None:
        """
        Train the CBOW model.

        Args:
            training_data (list): List of tuples (context_indices, target_index).
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for updates.
        """
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(training_data)  # Shuffle training data each epoch
            for context_indices, target_index in training_data:
                # Convert inputs to tensors
                context_indices_tensor = torch.tensor(context_indices, dtype=torch.long)
                # Forward pass
                scores = self.forward(context_indices_tensor)
                # Compute loss (negative log-likelihood)
                loss = -torch.log(self.softmax(scores)[target_index] + 1e-9)
                total_loss += loss.item()
                # Backward pass and update weights
                self.backward(target_index, scores, learning_rate)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")

    def get_embedding(self, word_index: int) -> Tensor:
        """
        Get the embedding vector for a given word index.

        Args:
            word_index (int): Index of the word in the vocabulary.

        Returns:
            Tensor: Embedding vector of the word.
        """
        return self.embedding.weights[word_index]

    def __str__(self) -> str:
        return "CBOWModel"

# Example Usage:

# Assuming we have the following vocabulary and corpus
vocabulary = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Create training data
def generate_training_data(corpus: list, window_size: int) -> list:
    """
    Generate training data for CBOW model.

    Args:
        corpus (list): List of words in the corpus.
        window_size (int): Number of context words to consider on each side.

    Returns:
        list: List of tuples (context_indices, target_index).
    """
    data = []
    for i in range(window_size, len(corpus) - window_size):
        context = (
            corpus[i - window_size:i] + corpus[i + 1:i + window_size + 1]
        )
        target = corpus[i]
        context_indices = [word_to_index[word] for word in context]
        target_index = word_to_index[target]
        data.append((context_indices, target_index))
    return data

corpus = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
window_size = 2
training_data = generate_training_data(corpus, window_size)

# Initialize and train CBOW model
vocab_size = len(vocabulary)
embed_size = 5  # Embedding size can be set as needed
cbow_model = CBOWModel(vocab_size, embed_size)
cbow_model.train(training_data, epochs=100, learning_rate=0.01)

# Get embedding for a word
word = "fox"
word_index = word_to_index[word]
embedding_vector = cbow_model.get_embedding(word_index)
print(f"Embedding for '{word}': {embedding_vector}")