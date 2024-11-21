# train_gpt2.py
import torch
from mlgroup1.scripts.gpt2 import GPT2
from nn_objects2 import CrossEntropyLoss, AdamOptim
from mlgroup1.scripts.utils2 import generate_square_subsequent_mask
from torch.utils.data import Dataset, DataLoader
from typing import Callable

# 데이터셋 정의
class TextDataset(Dataset):
    def __init__(self, texts: list, tokenizer: Callable, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        for text in texts:
            tokens = tokenizer(text)
            tokens = tokens[:max_length]
            tokens += [0] * (max_length - len(tokens))  # 패딩 (0으로 가정)
            self.inputs.append(tokens[:-1])
            self.targets.append(tokens[1:])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

# 간단한 토크나이저 예제
def simple_tokenizer(text: str) -> list:
    return [ord(c) for c in text]  # 문자 아스키 코드 사용

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    vocab_size = 256  # 아스키 코드 범위
    embed_size = 128
    max_seq_len = 50
    num_layers = 2
    heads = 4
    forward_expansion = 4
    dropout = 0.1
    learning_rate = 1e-3
    epochs = 10
    batch_size = 16

    # 데이터 준비
    texts = [
        "Hello, how are you?",
        "I am fine, thank you!",
        "What about you?",
        "I am doing well.",
        "This is a simple GPT-2 implementation.",
        "Let's train it on some text data."
    ]
    dataset = TextDataset(texts, simple_tokenizer, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    model = GPT2(vocab_size, embed_size, max_seq_len, num_layers, heads, forward_expansion, dropout)
    criterion = CrossEntropyLoss()
    optimizer = AdamOptim(lr=learning_rate)

    # 학습 루프
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            # 마스크 생성
            mask = generate_square_subsequent_mask(batch_inputs.size(1))

            # 순전파
            outputs = model.forward(batch_inputs, mask)  # [batch_size, seq_length, vocab_size]

            # 손실 계산
            loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))

            # 역전파
            grad = criterion.gradient(outputs.view(-1, vocab_size), batch_targets.view(-1))
            grad = grad.view(outputs.size())
            model.backward(grad)

            # 옵티마이저 스텝
            optimizer.step(model.transformer_blocks + [model.fc_out, model.embedding])  # 임베딩 레이어 추가

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
