##########
# ### Imports

# rust micrograd
from rust_nn import MLP, Value

##########
# ### Create Fake Data

# each observation has 3 input features
observations = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# expected output for each observation
targets = [1.0, -1.0, -1.0, 1.0]

##########
# ### Create Neural Network

# 3 inputs, 1 hidden layer with 4 neurons, and 1 output neuron
model = MLP(3, [4, 1])

##########
# ### Train Neural Network

# train for 10 epochs
for epoch in range(10):

    # calculate predictions and loss
    predictions = [model(observation) for observation in observations]
    loss: Value = sum(  # type: ignore
        (prediction - target) ** 2 for prediction, target in zip(predictions, targets)
    )

    # calculate gradients for every parameter
    model.zero_grad()
    loss.backward()

    # update every parameter
    for parameter in model.parameters():
        parameter.data += -0.01 * parameter.grad

    print(f"Epoch {epoch + 1}, Loss: {loss.data}")
