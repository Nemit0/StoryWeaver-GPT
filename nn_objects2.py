# nn_objects.py
import torch
import abc
from torch import tensor, Tensor
from tqdm import tqdm
from typing import Callable, Dict

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

### Helper Classes and Functions ###
class OneHotEncoder:
    def __init__(self, num_classes: int, data: Tensor) -> None:
        self.num_classes: int = num_classes
        self.token_map: Dict[any, int] = {token: i for i, token in enumerate(data.unique())}
        self.inv_map: Dict[int, any] = {v : k for k,v in self.token_map.items()}
    
    def encode(self, data: Tensor) -> Tensor:
        encoded = torch.zeros(data.shape[0], self.num_classes)
        for i, token in enumerate(data):
            encoded[i][self.token_map[token]] = 1
        return encoded

### LOSS FUNCTIONS ###
class Loss(abc.ABC):
    @abc.abstractmethod
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def gradient(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pass

class MeanSquaredError(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return torch.mean((y_pred - y_true) ** 2)

    def __str__(self) -> str:
        return "MeanSquaredError"

    def gradient(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return 2 * (y_pred - y_true) / y_true.shape[0]

class CrossEntropyLoss(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = torch.log_softmax(y_pred, dim=1)
        return -torch.sum(y_pred[range(y_true.shape[0]), y_true]) / y_true.shape[0]

    def __str__(self) -> str:
        return "CrossEntropyLoss"

    def gradient(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred[range(y_true.shape[0]), y_true] -= 1
        return y_pred / y_true.shape[0]

### ACTIVATION FUNCTIONS ###
class Activation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def gradient(self, x: Tensor) -> Tensor:
        pass

class Tanh(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        return torch.tanh(x)

    def __str__(self) -> str:
        return "Tanh"

    def gradient(self, x: Tensor) -> Tensor:
        y = self.__call__(x)
        return 1 - y ** 2

class Sigmoid(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))

    def __str__(self) -> str:
        return "Sigmoid"

    def gradient(self, x: Tensor) -> Tensor:
        y = self.__call__(x)
        return y * (1 - y)

class ReLU(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        return x * (x > 0).float()

    def __str__(self) -> str:
        return "ReLU"

    def gradient(self, x: Tensor) -> Tensor:
        return (x > 0).float()

class Softmax(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        exps = torch.exp(x - torch.max(x, axis=1, keepdim=True).values)
        return exps / torch.sum(exps, axis=1, keepdim=True)

    def __str__(self) -> str:
        return "Softmax"

    def gradient(self, x: Tensor) -> Tensor:
        return self.__call__(x) * (1 - self.__call__(x))

class Linear(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        return x

    def __str__(self) -> str:
        return "Linear"

    def gradient(self, x: Tensor) -> Tensor:
        return torch.ones_like(x)

### OPTIMIZERS ###
class Optimizer(abc.ABC):
    @abc.abstractmethod
    def step(self, layers: list) -> None:
        pass

class SimpleGradientDescent(Optimizer):
    def __init__(self, lr: float):
        self.lr: float = lr

    def step(self, layers: list) -> None:
        for layer in layers:
            layer.weights -= self.lr * layer.grad_weights
            layer.biases -= self.lr * layer.grad_biases

class GradientDescentWithMomentum(Optimizer):
    """
    Generated with Chatgpt for optimizer implementation example
    """
    def __init__(self, lr: float, momentum: float = 0.9):
        self.lr: float = lr
        self.momentum: float = momentum
        self.velocities: Dict[int, Dict[str, Tensor]] = {}

    def step(self, layers: list) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.velocities:
                self.velocities[idx] = {
                    'weights': torch.zeros_like(layer.weights),
                    'biases': torch.zeros_like(layer.biases)
                }
            v_w = self.velocities[idx]['weights']
            v_b = self.velocities[idx]['biases']
            v_w = self.momentum * v_w - self.lr * layer.grad_weights
            v_b = self.momentum * v_b - self.lr * layer.grad_biases
            layer.weights += v_w
            layer.biases += v_b
            self.velocities[idx]['weights'] = v_w
            self.velocities[idx]['biases'] = v_b

class AdamOptim(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}
    
    def step(self, layers: list) -> None:
        self.t += 1
        for idx, layer in enumerate(layers):
            # Initialize moment vectors if not present
            if idx not in self.m:
                self.m[idx] = {'weights': torch.zeros_like(layer.weights), 'biases': torch.zeros_like(layer.biases)}
                self.v[idx] = {'weights': torch.zeros_like(layer.weights), 'biases': torch.zeros_like(layer.biases)}
            
            # Update first moment
            self.m[idx]['weights'] = self.beta1 * self.m[idx]['weights'] + (1 - self.beta1) * layer.grad_weights
            self.m[idx]['biases'] = self.beta1 * self.m[idx]['biases'] + (1 - self.beta1) * layer.grad_biases
            
            # Update second moment
            self.v[idx]['weights'] = self.beta2 * self.v[idx]['weights'] + (1 - self.beta2) * (layer.grad_weights ** 2)
            self.v[idx]['biases'] = self.beta2 * self.v[idx]['biases'] + (1 - self.beta2) * (layer.grad_biases ** 2)
            
            # Bias-corrected moments
            m_hat_w = self.m[idx]['weights'] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m[idx]['biases'] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v[idx]['weights'] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v[idx]['biases'] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            layer.weights -= self.lr * m_hat_w / (torch.sqrt(v_hat_w) + self.epsilon)
            layer.biases -= self.lr * m_hat_b / (torch.sqrt(v_hat_b) + self.epsilon)

### LAYER AND NETWORK ###
class Layer:
    def __init__(self, input_size: int, output_size: int, activation: Activation) -> None:
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.activation: Activation = activation
        self.weights: Tensor = torch.randn(input_size, output_size, requires_grad=False) * 0.01
        self.biases: Tensor = torch.randn(output_size, requires_grad=False) * 0.01

    def forward(self, inputs: Tensor) -> Tensor:
        """
        3D 입력 텐서 [batch_size, seq_length, input_size]를 2D 텐서로 변환하여 선형 변환을 수행한 후,
        다시 3D 텐서로 변환하여 반환합니다.
        """
        batch_size, seq_length, _ = inputs.shape
        self.inputs = inputs  # [batch_size, seq_length, input_size]
        # 3D 텐서를 2D 텐서로 변환
        self.inputs_reshaped = inputs.view(batch_size * seq_length, self.input_size)  # [batch_size*seq_length, input_size]
        # 선형 변환
        z = torch.mm(self.inputs_reshaped, self.weights) + self.biases  # [batch_size*seq_length, output_size]
        # 다시 3D 텐서로 변환
        self.z = z.view(batch_size, seq_length, self.output_size)  # [batch_size, seq_length, output_size]
        # 활성화 함수 적용
        self.output = self.activation(self.z)
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        3D 텐서로 전달된 grad_output을 2D 텐서로 변환하여 역전파를 수행한 후,
        다시 3D 텐서로 변환하여 반환합니다.
        """
        batch_size, seq_length, _ = grad_output.shape
        # 3D 텐서를 2D 텐서로 변환
        grad_output_reshaped = grad_output.view(batch_size * seq_length, self.output_size)  # [batch_size*seq_length, output_size]
        # 활성화 함수의 그래디언트 계산
        grad_activation = self.activation.gradient(self.z).view(batch_size * seq_length, self.output_size)
        # 최종 그래디언트 계산
        grad = grad_output_reshaped * grad_activation  # [batch_size*seq_length, output_size]
        # 가중치와 편향에 대한 그래디언트 계산
        self.grad_weights = torch.mm(self.inputs_reshaped.t(), grad)  # [input_size, output_size]
        self.grad_biases = torch.sum(grad, dim=0)  # [output_size]
        # 입력에 대한 그래디언트 계산
        grad_input = torch.mm(grad, self.weights.t())  # [batch_size*seq_length, input_size]
        # 다시 3D 텐서로 변환
        grad_input = grad_input.view(batch_size, seq_length, self.input_size)  # [batch_size, seq_length, input_size]
        return grad_input

class NeuralNetwork:
    def __init__(self,
                 layer_sizes: list[int],
                 activation: Activation = Sigmoid(),
                 output_activation: Activation = None,
                 loss: Loss = MeanSquaredError(),
                 optimizer: Optimizer = None,
                 verbose: bool = False
                 ) -> None:
        self.layer_sizes: list = layer_sizes
        self.activation: Activation = activation
        self.output_activation: Activation = output_activation if output_activation else activation
        self.loss: Loss = loss
        self.optimizer: Optimizer = optimizer
        self.verbose: bool = verbose
        self.layers: list[Layer] = []
        num_layers: int = len(layer_sizes) - 1
        for i in range(num_layers):
            act = self.output_activation if i == num_layers - 1 else self.activation
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], act))

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def predict(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def backward(self, grad: Tensor) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self,
              inputs: Tensor,
              targets: Tensor,
              epochs: int,
              batch_size: int = 64
              ) -> Dict[str, Tensor]:
        history: Dict[str, Tensor] = {'loss': torch.zeros(epochs)}
        num_samples = inputs.shape[0]
        for epoch in tqdm(range(epochs)):
            permutation = torch.randperm(num_samples)
            epoch_loss = 0.0
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i+batch_size]
                batch_inputs = inputs[indices]
                batch_targets = targets[indices]
                outputs = self.forward(batch_inputs)
                loss = self.loss(outputs, batch_targets)
                epoch_loss += loss.item()
                grad = self.loss.gradient(outputs, batch_targets)
                self.backward(grad)
                if self.optimizer:
                    self.optimizer.step(self.layers)
            history['loss'][epoch] = epoch_loss / (num_samples // batch_size)
            if self.verbose and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        return history

    def __str__(self):
        return f"NeuralNetwork(layer_sizes={self.layer_sizes})\nActivation: {self.activation}"
