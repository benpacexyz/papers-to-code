##########
# ### Imports

# math
import math

##########
# ### Define Custom Classes for Micrograd

# value
class Value:

    def __init__(self, data, _children: tuple = (), _op: str = '', label: str = ''):
        self.data = data
        self.grad: float = 0.0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # defines console printing, alternative is object in mem address
    # a = Value(2.0)
    # a: in python console is equivalent to __repr__
    def __repr__(self):
        return f"Value(data={self.data})"

    # defines + operator for Value
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data + other.data,
            _children=(self, other),
            _op='+'
        )

        # e.g. if c = a + b, both gradients are 1
        # because increasing a or b by 1 increases c by 1
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    # defines * operator for Value
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data * other.data,
            _children=(self, other),
            _op='*'
        )

        # e.g. if c = a * b, the gradient of a is b
        # because increasing a by 1 increases c by b
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # defines ** operator for Value
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(
            data=self.data ** other,
            _children=(self,),
            _op=f'**{other}'
        )

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward

        return out

    # negative operator
    def __neg__(self):
        return self * -1

    # subtract operator
    def __sub__(self, other):
        return self + (-other)

    # division using trick a / b = a * (b**-1) = a * (1/b)
    def __truediv__(self, other):
        return self * other**-1

    # r methods that allow for python right object op e.g. 2 * a, a = Value(3)
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self**-1

    # tanh: function can be arbitrary complicated
    # as long as local differentiation is possible
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(
            data=t,
            _children=(self,),
            _op='tanh'
        )

        # local derivative of tanh(x) is 1 - tanh(x)^2
        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    # relu: returns 0 for negative values
    # and the value itself for positive values
    def relu(self):
        out = Value(
            data=max(0, self.data),
            _children=(self,),
            _op='ReLU'
        )

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    # runs backpropagation from this node through the whole graph
    def backward(self):

        # topological sort so each node's _backward runs
        # only after all nodes that depend on it
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            v, processed = stack.pop()

            if processed:
                topo.append(v)

            elif v not in visited:
                visited.add(v)
                stack.append((v, True))

                for child in v._prev:
                    stack.append((child, False))

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
