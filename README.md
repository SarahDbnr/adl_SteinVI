# Collaborative Work
This repository is a collaborative work by **Sarah Deubner**, **Kilian Runnwerth**, and **Luke-Liam Bergmeier**.

# Introduction
This repository employs a Stein Variational Inference (Stein VI) approach to apply Bayesian Inference via Variational Gradient Descent (VGD) for optimizing Bayesian Neural Networks (BNNs). The work is based on the ideas presented in the following paper: 
[Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471).

A detailed **Getting Started** guide, including a comprehensive setup manual and code explanation, is available in the project documentation. You can access it in HTML format.

To dive into the documentation:

1. Navigate to the `doc/build/index.html` file.
2. Open the file in your web browser.

This guide provides in-depth instructions on setting up the project, running examples, and understanding the codebase.

# Overview of Features and Capabilities

- **Bayesian Neural Networks (BNNs)**: Implements BNNs using Stein Variational Inference (Stein VI), providing probabilistic machine learning models with uncertainty estimation for tasks like regression and classification.
- **Stein Variational Gradient Descent (SVGD)**: Uses particle-based variational inference methods like SVGD and sSVGD to approximate the posterior distribution of model parameters efficiently.
- **Blackjax Integration**: Leverages the Blackjax library for implementing advanced variational inference algorithms and extending them to sSVGD, SVN, and sSVN.
- **Ready-to-Use Examples**: Includes various examples and scripts for easy implementation of BNNs on popular datasets like MNIST, FashionMNIST, and synthetic regression data.
- **JAX**: Uses the JAX library for fast and efficient automatic differentiation and accelerated linear algebra on CPUs, GPUs, and TPUs.

# Prerequisites
Before starting, ensure you have Python 3.10+, pip, and pipenv installed. All additional dependencies (e.g., JAX, NumPy, scikit-learn, BlackJAX) will be installed automatically.

# Setup

1. **Clone the Repository**  
   Download the repository to your local machine.

2. **Install Dependencies**  
   You can choose between two installation modes:
   - **Development Installation**: Installs core dependencies and tools for development, testing, and documentation `pipenv --dev install`
   - **Standard Installation**: Installs only the core dependencies to run the project `pipenv install`

3. **Activate the Environment**  
   Activate the virtual environment after installing dependencies.

4. **Set PYTHONPATH**  
   Ensure that `PYTHONPATH` is set to the root directory of the project for proper module resolution.

Once these steps are complete, your environment will be ready to use.


# Running Examples

To run a example, navigate to the `run_stein_vi/Example_Run.py` file and select the dataset you'd like to use. Place the relevant function call under the `if __name__ == "__main__":` block. For instance, to run the MNIST dataset example, use:

```python
if __name__ == "__main__":
    run_MNIST()
```

# Tests

To run the test suite, use pytest. This will execute all available unit tests and ensure the integrity of the code.

Run all tests with:
```bash
pytest tests
```



