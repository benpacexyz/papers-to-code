# Papers to Code

This repo is a collection of projects that explores and implements research papers and foundational concepts related to machine learning and neural networks.

The primary purpose of this repo is personal learning and to showcase some fun / interesting concepts. Each top-level directory is a mostly self-contained project with its own source code, dependencies, data requirements, and documentation.

## Projects

### Micrograd

[Micrograd](./micrograd/) is a small neural network and automatic differentiation implementation built for understanding backpropagation from the ground up. It includes a scalar valued computation engine, neural network components, and a simple multi-layer perceptron training example.

The project contains both a Python implementation and a Rust implementation exposed to Python through PyO3. This was done primarily to compare performance between the implementations, and to test the PyO3 integration between Python <> Rust.

### WebPhish

[WebPhish](./webphish/README.md) is an implementation inspired by the paper [Look before you leap: Detecting phishing web pages by exploiting raw URL and HTML characteristics](https://www.sciencedirect.com/science/article/pii/S0957417423016858) by Chidimma Opara, Yingke Chen, and Bo Wei.

The project uses a convolutional neural network to classify web pages from tokenized HTML. It includes a paper inspired baseline model, an optimized model developed through architecture and hyperparameter search, and separate training, testing, and final validation results.
