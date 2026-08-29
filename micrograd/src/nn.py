##########
# ### Imports

# standard
import random

# micrograd engine
from engine import Value

##########
# ### Define Custom Classes for Neural Network

# base class for neural network components
class Module:

    # can be called to zero out gradients
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> list[Value]:
        return []

# neuron which represents Activation_Function(sum(x * w) + b) = scalar value
class Neuron(Module):

    def __init__(self, nin: int, nonlin: bool = True) -> None:
        # initializes weights and bias for all input Value(s)
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

        # determines if non-linear activation is applied or not
        self.nonlin = nonlin

    # call usage example
    # __init__: n = Neuron(x=3)
    # __call__: output = n([x1, x2, x3])
    def __call__(self, x: list) -> Value:
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    # returns list of parameters (i.e. weights + bias)
    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    # defines console printing behavior
    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

# layer is a collection of neurons that receive the same input vector x
class Layer(Module):

    # layer is initialized by creating a vector of nout neurons
    def __init__(self, nin: int, nout: int, **kwargs: bool) -> None:
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    # calls __call__ for all neurons in layer and returns output
    def __call__(self, x: list) -> list[Value]:
        out = [n(x) for n in self.neurons]
        return out

    # calls parameters for all neurons in layer
    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


# multi-layer perceptron class
class MLP(Module):

    # construct the full multi-layer perceptron
    # nin = 3, means every obs has 3 features
    # nouts = [4, 4, 1], means 2 hidden layers with 4 neurons, and 1 output layer with 1 neuron
    # in this case 1 neuron could be a regression problem or binary classification
    def __init__(self, nin: int, nouts: list[int]) -> None:
        sizes = [nin] + nouts
        self.layers = []

        for i in range(len(nouts)):
            input_size = sizes[i]
            output_size = sizes[i + 1]
            is_last_layer = i == len(nouts) - 1

            self.layers.append(
                Layer(input_size, output_size, nonlin=not is_last_layer)
            )

    def __call__(self, x: list) -> Value | list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
