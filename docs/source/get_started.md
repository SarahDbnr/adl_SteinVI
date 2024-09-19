# Getting Started with Stein Variational Inference for Bayesian Neural Networks

This guide will help you set up and run Stein Variation Inference algorithms for Bayesian Neural Networks (BNNs). Our project leverages a particle-based variational inference method, to approximate Bayesian posterior distributions over the weights of neural networks.

Bayesian Neural Networks provide a probabilistic approach to machine learning, capturing model uncertainty in predictions, which is especially useful for tasks such as regression and classification. This repository provides implementations and extensions of Stein Variation Gradient Descent (SVGD) and Stein Variational Newton Method (SVN), based on the python library [Blackjax](https://blackjax-devs.github.io/blackjax/), for Bayesian Neural Networks.

## Collaborative Work
This repository is a collaborative effort between **Sarah Deubner**, **Kilian Runnwerth**, and **Luke-Liam Bergmeier**.

## Introduction
This repository employs Stein Variational Inference (Stein VI) to perform Bayesian Inference via Variational Gradient Descent (VGD) for optimizing Bayesian Neural Networks. The approach is rooted in the work described in the paper [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471), which lays the foundation for this particle-based approximation method.


## Prerequisites

Before you begin, ensure you have Python 3.10+ installed. All other dependencies, including JAX, NumPy, scikit-learn, BlackJAX, tqdm, and so on, will be installed automatically when you set up the environment.

## Setup

1. **Clone the Repository**  
   First, clone the repository to your local machine:
   ```
   git clone https://github.com/SarahDbnr/adl_SteinVI.git
   cd adl_SteinVI
   ```

2. **Install Dependencies**  
The project supports two installation modes:

- **Development Installation (`--dev`)**: Install both core dependencies and development tools (e.g., `pytest`, `sphinx`, `myst-parser`) by running:

  ```
  pipenv install --dev
  ```
  This is recommended if you plan to contribute to the codebase, work with documentation or run tests.

- **Standard Installation**: Install only the core dependencies needed to run the project without development tools by using:

  ```
  pipenv install
  ```

3. **Activate the Environment**  
Once the dependencies are installed, activate the environment by running:

  ```
  pipenv shell
  ```

4. **Set PYTHONPATH**  
  Ensure that `PYTHONPATH` is set to the root directory of the project for proper module resolution. You can add the following to your shell configuration (e.g., `.bashrc`, `.zshrc`):
  ```
  export PYTHONPATH=$(pwd)
  ```

## Project Structure
  ```
.
├── data/                                      # Contains data handling scripts and dataset information
│   ├── data_handling.py                       # Functions to handle datasets
│   ├── datasets_info.py                       # Contains metadata or functions to retrieve dataset info
│   └── regression_toy_example.py              # Example script for running regression tasks on toy datasets
├── model/                                     # Main folder for the Bayesian Neural Network (BNN) model implementation
│   ├── BNN_Model.py                           # Defines the Bayesian Neural Network class
│   ├── Example_Run_parameter_search.py        # Script for running parameter search experiments
│   ├── Example_Run.py                         # Script for running example training experiments
│   ├── Examples_Run_advanced_algorithms.py    # Advanced algorithms for running example BNNs
│   ├── stein_vi/                              # Implementation of Stein Variational Inference (SVI) algorithms
│   │   ├── algorithm/
│   │   │   ├── plain_SVGD.py                  # Implementation of plain Stein Variational Gradient Descent (SVGD)
│   │   │   ├── SVGD.py                        # Main SVGD implementation script
│   │   │   └── quasi_SVN_with_lbfgs.py        # Quasi-Stochastic Variational Newton with L-BFGS optimizer
│   │   ├── Classes/
│   │   │   ├── Handler_Class.py               # Class to handle operations like training and evaluation
│   │   │   ├── Parameter_Class.py             # Class for managing parameters
│   │   │   └── SteinVI_BNN.py                 # Core class for BNN using Stein Variational Gradient Descent
│   │   ├── metrics/
│   │   │   ├── plots_validation_metrics.py    # Script for plotting validation metrics
│   │   │   ├── validation_and_evaluation.py   # Contains functions for model evaluation
│   │   │   └── view_misclassified_images.py   # Helper function to view misclassified images
│   │   ├── svgd_plots/                        # Folder for visualizations related to SVGD training process
│   │   └── stein_vi.py                        # Script defining main training and evaluation loop for the BNN
│   └── run_stein_vi/                          # Scripts for running SVGD experiments
│       └── run_SVGD.py                        # Main entry point for running SVGD on datasets like MNIST
├── tests/                                     # Folder containing test scripts for the project
└── docs/                                      # Documentation folder
    ├── get_started.md                         # Guide for setting up and running the project
    └── index.rst                              # Sphinx documentation index file
  ```
## Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Set up your dataset. The code expects data in the following format:
   ```python
   dataset = (z_train, y_train, z_val, y_val, z_test, y_test)
   ```
   Where `z` represents input features and `y` represents target values or labels.

3. Create an instance of the `SteinVI_BNN` class (not shown in the provided code snippets, but assumed to exist):
   ```python
   from stein_vi import SteinVI_BNN
   
   steinvi = SteinVI_BNN(nnet_model, use_for_regression=True)  # or False for classification
   ```

4. Set up SVGD:
   ```python
   from svgd import set_up_svgd
   
   set_up_svgd(steinvi)
   ```

5. Train the model:
   ```python
   from model_training import train_general_algorithm
   import jax
   
   key = jax.random.PRNGKey(0)  # Set a random seed
   train_general_algorithm(steinvi, dataset, key)
   ```

6. Evaluate the model:
   ```python
   z_test, y_test = dataset[4], dataset[5]
   eval_metric1, eval_metric2, _ = steinvi.evaluate_fn(steinvi.state, z_test, y_test, print_out=True)
   ```

## Advanced Usage

### Mini-batching

The project supports different mini-batching modes for both data and particles. You can set these in the `steinvi.parameter` object:

- `batch_size`: Size of data mini-batches (0 for full batch)
- `particle_batch_size`: Size of particle mini-batches (0 for full batch)

### Early Stopping

Early stopping is implemented to prevent overfitting. You can configure it using:

- `steinvi.parameter.early_stopping`: Enable/disable early stopping
- `steinvi.parameter.patience_early_stopping`: Number of iterations to wait for improvement
- `steinvi.parameter.min_delta_early_stopping`: Minimum change to qualify as improvement

### Comparison with Random Forest

You can compare the performance of your SVGD-based Bayesian Neural Network with a Random Forest model:

```python
from random_forest import random_forest

rf_metrics = random_forest(dataset, task_type='regression')  # or 'classification'
print(rf_metrics)
```

## Contributing

We welcome contributions! Please see our `CONTRIBUTING.md` file (if available) for guidelines on how to submit issues, feature requests, and pull requests.

## License

[Include information about your project's license here]