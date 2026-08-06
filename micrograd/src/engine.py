##########
# ### Imports



##########
# ### Define Custom Classes for Micrograd

# value
class Value:

    def __init__(self, data, _children: tuple = (), _op: str = ''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # defines console printing, alternative is object in mem address
    # a = Value(2.0)
    # a: in python console is equivalent to __repr__
    def __repr__(self):
        return f"Value(data={self.data})"

    # defines + operator for Value
    def __add__(self, other):
        out = Value(
            data=self.data + other.data,
            _children=(self, other),
            _op='+'
        )
        return out

    # defines * operator for Value
    def __mul__(self, other):
        out = Value(
            data=self.data * other.data,
            _children=(self, other),
            _op='*'
        )
        return out

