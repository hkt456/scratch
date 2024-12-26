# From Scratch

This repository contains my implementation of different elements in AI using PyTorch from scratch (as much as possible).

> Currently working on activation functions and machine learning models

## Project Structure

| Type | Implementation | Description | Status |
|------|---------------|-------------|---------|
| Activation Functions | [`ReLU`](activation/relu/relu.py) | ReLU activation function implementation | ✅ |
| Activation Functions | [`Sigmoid`](activation/sigmoid/sigmoid.py) | Sigmoid activation function implementation | ✅ |
| Models | [`ML/GaussianNaiveBayes.py`](ML/GaussianNaiveBayes.py) | Gaussian Naive Bayes model implementation | ✅ |
| Models | [`ML/KNearestNeighbors.py`](ML/KNearestNeighbors.py) | KKN model implementation | ✅ |
| Models | [`ML/LinearRegression.py`](ML/LinearRegression.py) | Logistic regression model implementation | ⏳ |


## Activation Functions

PyTorch provides several activation functions that you can use in your neural network implementations. Here are some of the commonly used ones:

- **ReLU (Rectified Linear Unit)**: `torch.nn.ReLU`
- **Sigmoid**: `torch.nn.Sigmoid`
- **Tanh (Hyperbolic Tangent)**: `torch.nn.Tanh`
- **Leaky ReLU**: `torch.nn.LeakyReLU`
- **Softmax**: `torch.nn.Softmax`
- **Softplus**: `torch.nn.Softplus`
- **ELU (Exponential Linear Unit)**: `torch.nn.ELU`
- **SELU (Scaled Exponential Linear Unit)**: `torch.nn.SELU`
- **GELU (Gaussian Error Linear Unit)**: `torch.nn.GELU`
- **Swish**: `torch.nn.SiLU` (also known as Sigmoid Linear Unit)

You can find more activation functions and their details in the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).

## Tribute

I would like to acknowledge the following sources for their invaluable tutorials and lessons that have greatly contributed to my understanding and implementation of AI:

- [oniani/ai](https://github.com/oniani/ai/tree/main)
- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.