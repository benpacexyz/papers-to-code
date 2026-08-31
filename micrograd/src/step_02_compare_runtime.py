##########
# ### Imports

# standard
import statistics
import time

# python micrograd
from nn import MLP as PythonMLP

# rust micrograd
from rust_nn import MLP as RustMLP

##########
# ### Benchmark Setup

# number of times to train each model
BENCHMARK_RUNS = 5
EPOCHS = 250
LEARNING_RATE = .01

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
# ### Define Benchmark Functions

# train a model and return its runtime in seconds
def train_model(model) -> float:
    start_time = time.perf_counter()

    for _ in range(EPOCHS):

        # calculate predictions and loss
        predictions = [model(observation) for observation in observations]
        loss = sum(  # type: ignore
            (prediction - target) ** 2 for prediction, target in zip(predictions, targets)
        )

        # calculate gradients for every parameter
        model.zero_grad()
        loss.backward()

        # update every parameter
        for parameter in model.parameters():
            parameter.data += -LEARNING_RATE * parameter.grad

    return time.perf_counter() - start_time


# run the benchmark multiple times with a new model each run
def benchmark_model(model_class) -> list[float]:
    return [
        train_model(model_class(3, [4, 1]))
        for _ in range(BENCHMARK_RUNS)
    ]


# print summary statistics for one implementation
def print_runtime_results(name: str, runtimes: list[float]) -> None:
    print('{0} Micrograd Results:'.format(name))
    print('  Mean Runtime: {0:.4f} seconds'.format(statistics.mean(runtimes)))
    print('  Minimum Runtime: {0:.4f} seconds'.format(min(runtimes)))
    print('  Maximum Runtime: {0:.4f} seconds'.format(max(runtimes)))
    print()


##########
# ### Compare Python and Rust Runtime

# benchmark both implementations using the same training workload
python_runtimes = benchmark_model(PythonMLP)
rust_runtimes = benchmark_model(RustMLP)

# print individual runtime results
print_runtime_results('Python', python_runtimes)
print_runtime_results('Rust', rust_runtimes)

# calculate which implementation was faster
python_mean_runtime = statistics.mean(python_runtimes)
rust_mean_runtime = statistics.mean(rust_runtimes)

if python_mean_runtime < rust_mean_runtime:
    faster_implementation = 'Python'
    speedup = rust_mean_runtime / python_mean_runtime
else:
    faster_implementation = 'Rust'
    speedup = python_mean_runtime / rust_mean_runtime

# print final comparison
print('Runtime Comparison:')
print('  Faster Implementation: {0}'.format(faster_implementation))
print('  Speedup: {0:.2f}x'.format(speedup))
# Python Mean Runtime: 0.0827 seconds
# Rust Mean Runtime:   0.0111 seconds
# Rust Speedup:        7.45x
