import torch
import abc
from torch import tensor, Tensor
from tqdm import tqdm
from typing import Callable, Dict

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

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
        # Apply softmax to get probabilities
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

### LAYER AND NETWORK ###
class Layer:
    def __init__(self, input_size: int, output_size: int, activation: Activation) -> None:
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.activation: Activation = activation
        self.weights: Tensor = torch.rand(input_size, output_size)
        self.biases: Tensor = torch.rand(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs: Tensor = inputs
        self.z: Tensor = torch.mm(inputs, self.weights) + self.biases
        self.output: Tensor = self.activation(self.z)
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        grad_activation = self.activation.gradient(self.z)
        grad = grad_output * grad_activation
        self.grad_weights = torch.mm(self.inputs.t(), grad)
        self.grad_biases = torch.sum(grad, axis=0)
        grad_input = torch.mm(grad, self.weights.t())
        return grad_input

class NeuralNetwork:
    def __init__(self,
                 layer_sizes: list,
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
                self.optimizer.step(self.layers)
            history['loss'][epoch] = epoch_loss / (num_samples // batch_size)
        return history

    def __repr__(self):
        return f"NeuralNetwork(layer_sizes={self.layer_sizes})\nActivation: {self.activation}"

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch import sin, cos

    layer_sizes = [2, 10, 1]
    optimizer = SimpleGradientDescent(lr=0.01)
    nn = NeuralNetwork(layer_sizes, 
                       activation=Tanh(), 
                       optimizer=optimizer)
    print(nn.layers)

    x = torch.linspace(-10, 10, 100).view(-1, 1)
    y = torch.linspace(-10, 10, 100).view(-1, 1)
    inputs = torch.cat([x, y], dim=1)
    targets = sin(x) + cos(y)
    history = nn.train(inputs, targets, epochs=1000)
    print(history['loss'][-1])
    print(nn.forward(inputs))
    plt.plot(history['loss'].cpu().numpy())
    plt.show()
    plt.savefig('loss.png')