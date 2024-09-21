# Getting Started with Stein Variational Inference for Bayesian Neural Networks

## Table of Contents
1. [Introduction](introduction)
2. [Collaborative Work](collaborative-work)
3. [Prerequisites](prerequisites)
4. [Setup](setup)
5. [Project Structure](project-structure)
6. [Brief Overview of Algorithms](brief-overview-of-algorithms)
7. [Running the Examples](running-the-examples)
8. [Code Structure: Training and Evaluation](code-structure-training-and-evaluation)
9. [Understanding the Main Classes](understanding-the-main-classes)
10. [Collected Findings](collected-findings)
11. [Further Reading](further-reading)

(introduction)=
## Introduction

This guide will help you set up and run Stein Variation Inference algorithms for Bayesian Neural Networks (BNNs). Our project leverages a particle-based variational inference method, to approximate Bayesian posterior distributions over the weights of neural networks.

Bayesian Neural Networks provide a probabilistic approach to machine learning, capturing model uncertainty in predictions, which is especially useful for tasks such as regression and classification. This repository provides implementations and extensions of Stein Variation Gradient Descent (SVGD) and Stein Variational Newton Method (SVN), based on the python library [Blackjax](https://blackjax-devs.github.io/blackjax/), for BNNs.

This repository employs Stein Variational Inference (Stein VI) to perform Bayesian Inference via Variational Gradient Descent (VGD) for optimizing Bayesian Neural Networks. The approach is rooted in the work described in the paper [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471), which lays the foundation for this particle-based approximation method.

(collaborative-work)=
## Collaborative Work
This repository is a collaborative effort between **Sarah Deubner**, **Kilian Runnwerth**, and **Luke-Liam Bergmeier**.

(prerequisites)=
## Prerequisites
Before you begin, ensure you have Python 3.10+ installed. All other dependencies, including JAX, NumPy, scikit-learn, BlackJAX, tqdm, and so on, will be installed automatically when you set up the environment.

(setup)=
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

(project-structure)=
## Project Structure

The `stein_vi` directory contains the core implementation for training Bayesian Neural Networks (BNNs) using Stein Variational Gradient Descent (SVGD), stochastic Stein Variational Gradient Descent (sSVGD) and Stein Variantional Newton method (SVN). This package encapsulates all the logic and components necessary for performing variational inference on BNNs. It includes algorithms, helper classes, metrics for evaluation, parameter management and plotting functions.

```
.
stein_vi/
├── algorithm/
│   ├── quasiSVN/
│   │   ├── local_blackjax_file_adjusted_for_lbfgs.py
│   │   └── quasiSVN.py
│   ├── sSVGD/
│   │   ├── local_blackjax_file_with_adjustments_for_sSVGD.py
│   │   ├── matrices_for_noise_matrix.py
│   │   ├── ssvgd.py
│   │   └── test_matrix_equivalent_to_tests_later.py
│   ├── get_posteriori.py
│   ├── model_training.py
│   ├── random_forest.py
│   └── svgd.py
├── Classes/
│   ├── Handler_Class.py
│   ├── Parameter_Class.py
│   └── SteinVI_BNN_Class.py
├── metrics/
│   ├── plots_validation_metrics.py
│   ├── validation_and_evaluation.py
│   └── view_misclassified_images.py
├── parameter_search/
│   └── print_evaluation.py
└── stein_vi.py
```

The `run_stein_vi` directory provides ready-to-run examples that use the core logic from the `stein_vi` package. These scripts demonstrate how to use the SVGD, sSVGD and SVN-based Bayesian Neural Networks for specific tasks, including regression and classification. The examples are written to simplify running different configurations and datasets with minimal setup.

```
.
run_stein_vi/
├── data/
│   ├── data_handling.py
│   ├── datasets_info.py
│   └── regression_toy_example.py
├── model/
│   └── BNN_Model.py
├── Example_Run.py
├── Example_Run_advanced_algorithms.py
└── Example_Run_parameter_search.py
```

(brief-overview-of-algorithms)=
## Brief Overview of Algorithms

In this project, we employ several Stein Variational Inference (SVI) algorithms to optimize Bayesian Neural Networks (BNNs). Each algorithm is implemented using the `blackjax` library and integrates with the `SteinVI_BNN` class for managing training and evaluation processes.

### Stein Variational Gradient Descent (SVGD)
- **Source**: Based on the paper [Stein Variational Gradient Descent](https://arxiv.org/abs/1608.04471), SVGD is a particle-based variational inference method.
- **What it does**: It updates particles (which represent the model parameters) by iteratively moving them towards the Bayesian posterior distribution using gradients.
- **Implementation**: In our project, SVGD is implemented using `blackjax.vi.svgd` and the Radial Basis Function (RBF) kernel. The training process is managed by the `SteinVI_BNN` class through a state that tracks particles and kernel parameters.

### Stochastic Stein Variational Gradient Descent (sSVGD)
- **Source**: sSVGD builds upon SVGD by introducing stochasticity, making it more suitable for larger datasets or cases where full-batch updates are computationally expensive.
- **What it does**: Similar to SVGD, sSVGD updates particles using gradients, but with added stochastic noise to make the updates more varied. This allows for more efficient exploration of the posterior distribution.
- **Implementation**: sSVGD is implemented with some modifications to the `blackjax` framework to adjust the noise term for better performance. The noise is scaled appropriately to match the magnitude of the particles.

### Stein Variational Newton Method (SVN)
- **Source**: The Stein Variational Newton method builds on SVGD by approximating second-order derivatives, allowing for faster convergence in certain cases.
- **What it does**: SVN aims to improve the efficiency of particle updates by using second-order information (like Newton’s method) to refine the gradient updates.
- **Implementation**: This is integrated using a modified version of `blackjax` with the `LBFGS` optimizer from `optax`, making it more suitable for optimization problems with fewer iterations but higher computational complexity.

### Summary of Usage
- **SVGD**: General-purpose, particle-based inference method, efficient for most tasks.
- **sSVGD**: Adds stochastic noise to SVGD, particularly useful for larger datasets and models.
- **SVN**: A second-order method that can converge faster for certain problems but at a higher computational cost.

These algorithms are used interchangeably within the `SteinVI_BNN` class to manage the training of BNNs. You can configure the specific algorithm you want to use in the setup phase by calling the corresponding setup functions (`set_up_svgd`, `set_up_ssvgd`, or `set_up_quasi_SVN`).


(running-the-examples)=
## Running the Examples

(code-structure-training-and-evaluation)=
## Code Structure: Training and Evaluation

(understanding-the-main-classes)=
## Understanding the Main Classes

### Advanced Usage

#### Mini-batching

The project supports different mini-batching modes for both data and particles. You can set these in the `steinvi.parameter` object:

- `batch_size`: Size of data mini-batches (0 for full batch)
- `particle_batch_size`: Size of particle mini-batches (0 for full batch)

#### Early Stopping

Early stopping is implemented to prevent overfitting. You can configure it using:

- `steinvi.parameter.early_stopping`: Enable/disable early stopping
- `steinvi.parameter.patience_early_stopping`: Number of iterations to wait for improvement
- `steinvi.parameter.min_delta_early_stopping`: Minimum change to qualify as improvement

#### Comparison with Random Forest

You can compare the performance of your SVGD-based Bayesian Neural Network with a Random Forest model:

```python
from random_forest import random_forest

rf_metrics = random_forest(dataset, task_type='regression')  # or 'classification'
print(rf_metrics)
```

(collected-findings)=
## Collected Findings

(further-reading)=
## Further Reading