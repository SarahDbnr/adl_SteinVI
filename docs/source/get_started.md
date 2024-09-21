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

(introduction)=
## Introduction

This guide will help you set up and run Stein Variation Inference algorithms for Bayesian Neural Networks (BNNs). Our project leverages a particle-based variational inference method, to approximate Bayesian posterior distributions over the weights of neural networks.

Bayesian Neural Networks provide a probabilistic approach to machine learning, capturing model uncertainty in predictions, which is especially useful for tasks such as regression and classification. This repository provides implementations and extensions of Stein Variation Gradient Descent (SVGD) based on the python library [Blackjax](https://blackjax-devs.github.io/blackjax/), for BNNs.

This repository employs Stein Variational Inference (Stein VI) to perform Bayesian Inference via Variational Gradient Descent (VGD) for optimizing Bayesian Neural Networks. The approach is rooted in the work described in the paper [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471), which lays the foundation for this particle-based approximation method.

(collaborative-work)=
## Collaborative Work
This repository is a collaborative effort between **Sarah Deubner**, **Kilian Runnwerth**, and **Luke-Liam Bergmeier**.

(prerequisites)=
## Prerequisites
Before you begin, ensure you have Python 3.10+, pip and pipenv installed. All other dependencies, including JAX, NumPy, scikit-learn, BlackJAX, tqdm and so on, will be installed automatically when you set up the environment.

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

The `stein_vi` directory contains the core implementation for training Bayesian Neural Networks (BNNs) using Stein Variational Gradient Descent (SVGD) and stochastic Stein Variational Gradient Descent (sSVGD). This package encapsulates all the logic and components necessary for performing variational inference on BNNs. It includes algorithms, helper classes, metrics for evaluation, parameter management and plotting functions.

```
.
stein_vi/
├── algorithm/
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

The `run_stein_vi` directory provides ready-to-run examples that use the core logic from the `stein_vi` package. These scripts demonstrate how to use the SVGD and sSVGD based Bayesian Neural Networks for specific tasks, including regression and classification. The examples are written to simplify running different configurations and datasets with minimal setup.

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

### Summary of Usage
- **SVGD**: General-purpose, particle-based inference method, efficient for most tasks.
- **sSVGD**: Adds stochastic noise to SVGD, particularly useful for larger datasets and models.

These algorithms are used interchangeably within the `SteinVI_BNN` class to manage the training of BNNs. You can configure the specific algorithm you want to use in the setup phase by calling the corresponding setup functions (`set_up_svgd` or `set_up_ssvgd`).


(running-the-examples)=
## Running the Examples

We provide several examples to demonstrate how to run Stein Variational Inference (SVI) on different datasets using various algorithms. You can find these examples in the repository under the following files:

1. [run_stein_vi/Example_Run.py](example-run-py)  
2. [run_stein_vi/Example_Run_advanced_algorithms.py](example-run-advanced-algorithms-py)  
3. [run_stein_vi/Example_Run_parameter_search.py](example-run-parameter-search-py)

Please note that these are just a few of the available examples. The repository contains additional examples that cover various tasks, but only these three are highlighted here.


(example-run-py)=
### Example_Run.py

This script contains several examples of running **SVGD** with the Adaptive Moment Estimation (ADAM) optimizer on different datasets, both for regression and classification tasks. You need to place the specific function you'd like to run under the `if __name__ == "__main__":` block at the bottom of the file.

#### Available Runs:

- **Regression Toy Example**  
  A synthetic regression dataset is used to demonstrate SVGD:
  
  ```python
  if __name__ == "__main__":
      run_regression_toy_example()
  ```

- **MNIST Classification**  
SVGD is applied to the MNIST dataset for image classification:
  
  ```python
  if __name__ == "__main__":
      run_MNIST()
  ```

- **FashionMNIST Classification**  
SVGD is used on FashionMNIST for image classification tasks:
  
  ```python
  if __name__ == "__main__":
      run_FashionMNIST()
  ```

**Command to Exceute:**
   ```
   python run_stein_vi/Example_Run.py
   ```

(example-run-advanced-algorithms-py)=
### Example_Run_advanced_algorithms.py

This file showcases more advanced algorithms, such as **sSVGD** as well as **SVGD**, using different optimization techniques such as Gradient Descent (GD) and Limited Broyden–Fletcher–Goldfarb–Shanno (LBFGS) and ADAM.


#### Available Runs:

- **MNIST with Gradient Descent (GD)**  
Runs MNIST classification with SVGD using plain gradient descent:
  ```python
   if __name__ == "__main__":
      run_MNIST_GD()
   ```

- **MNIST with sSVGD**  
Runs MNIST classification with stochastic Stein Variational Gradient Descent (sSVGD):
  ```python
   if __name__ == "__main__":
      run_MNIST_ssvgd()
   ```

**Command to Exceute:**
   ```
   python run_stein_vi/Example_Run_advanced_algorithms.py
   ```

(example-run-parameter-search-py)=
### Example_Run_parameter_search.py

This example demonstrates parameter search for both regression and classification tasks. The script iterates over various parameters (e.g., number of particles, batch size, learning rate) and logs the results.
Running this file works similarly to the examples shown above. You simply choose the dataset and function you want to run and place it under the main execution block.

(code-structure-training-and-evaluation)=
## Code Structure: Training and Evaluation

When you run the code to train a Bayesian Neural Network (BNN) using Stein Variational Inference (SVI), several modules collaborate to handle different aspects of the process. Specifically, `SteinVI_BNN_Class.py`, `model_training.py` and the algorithm you choose for example `svgd.py` play crucial roles in setting up the model, initializing the SVGD algorithm, and executing the training loop.

Below is an overview of how these files interact during execution, with code examples to illustrate the workflow.

1. **Model and Training Setup:**
   In your main script or function (e.g., `run_MNIST`), you set up the model, data, and training parameters. This involves creating an instance of the `SteinVI_BNN` class from `SteinVI_BNN_Class.py`.

   The data must be structured as a tuple in the following order: `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test`.

   **Example:**
     ```python
   # Import necessary modules
   from stein_vi.Classes.SteinVI_BNN_Class import SteinVI_BNN
   from stein_vi.stein_vi import train_with_stein_vi
   from optax import adam, exponential_decay
   import jax

   # Initialize random key
   key = jax.random.PRNGKey(1)

   # Load and preprocess data, assuming the mnist dataset has the desired tuple structure
   z_train, _, _, _, _, _ = mnist_dataset

   # Define optimizer with exponential decay schedule
   optimizer = adam(
      exponential_decay(
         init_value=0.05,
         transition_steps=100,
         decay_rate=0.95,
         staircase=True
      )
   )

   # Build neural network model (assuming build_model is defined)
   nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

   # Create an instance of SteinVI_BNN
   steinvi_svdg = SteinVI_BNN(
      key=key,
      x_train=z_train,
      nnet=nnet_model,
      use_for_regression=False,
      optimizer=optimizer,
      batch_size=300,
      num_particles=50
   )

   ```

2. **Model and Training Setup:**
   You initiate the training process by calling `train_with_stein_vi` from `stein_vi.py`, specifying the SVI algorithm (e.g., SVGD).

   ```python
   # Start training using the SVGD algorithm
   train_with_stein_vi(
      steinvi=steinvi_svdg,
      dataset=mnist_dataset,
      key=key,
      algorithm="svgd"
   )
   ```
3. **Algorithm Configuration:**
   Within `train_with_stein_vi`, the appropriate setup function is called to configure the algorithm. For SVGD, `set_up_svgd` from `svgd.py` is used.
   
   In `stein_vi.py`:
   ```python
   def train_with_stein_vi(steinvi, dataset, key, algorithm="svgd"): 
      if algorithm == "svgd": set_up_svgd(steinvi) 
      # Additional algorithm setups... 

      # Proceed to the training loop 
      train_general_algorithm(steinvi, dataset, key) 
   ```
   In `svgd.py`:
   ```python
   def set_up_svgd(steinvi_svdg): 
      # Initialize SVGD state 
      steinvi_svdg.state, svgd = initialize_svgd_state(steinvi_svdg)
      
      # Define the update function for SVGD
      def svgd_update_fn(state, z_batch, y_batch, particle_indices=None):
         # Update particles using SVGD step

      # Attach the update function to SteinVI_BNN instance
      steinvi_svdg.update_fn = svgd_update_fn

      # Define the evaluation function
      def evaluate_model_fn(state, z_val, y_val, print_out):
         # Evaluate model performance
         # ...

      # Attach the evaluation function
      steinvi_svdg.evaluate_fn = evaluate_model_fn
   ````

   State Initialization `initialize_svgd_state` sets up the initial particles and kernel parameters, and function attachments `update_fn` and `evaluate_fn` are added to the `SteinVI_BNN` instance for use during training.

4. **Training Loop Execution:**
   `train_general_algorithm` from `model_training.py` is invoked to start the training loop based on the (particle-) minibatching configuration.

   In `stein_vi.py`:
   ```python
   train_general_algorithm(steinvi=steinvi, dataset=dataset, key=key) 
   ```
   In `model_training.py`:
   ```python
   def train_general_algorithm(steinvi, dataset, key): 
      # Determine the appropriate training loop based on batch sizes 
      if steinvi.parameter.batch_size != 0 and steinvi.parameter.particle_batch_size != 0: 
         training_loop_fn = data_and_particle_minibatch_training_loop 
      # Other conditions for different batch modes...
      
      # Start the selected training loop
      training_loop_fn(steinvi, dataset, key)
   ```

5. **Model Updates:**
   During each iteration, the `update_fn` defined in `svgd.py` updates the models particles.

   In `model_training.py` (within the selected training loop):

   ```python
   for iteration in range(steinvi.parameter.num_iterations): 
      # Create mini-batches of data and particles 
      z_train_batched, y_train_batched = create_minibatches(...) particle_indices_batches = create_particle_minibatch_indices(...) 
      for z_batch, y_batch in zip(z_train_batched, y_train_batched):
         for particle_indices in particle_indices_batches:
            # Update the models state (particles)
            steinvi.state = steinvi.update_fn(
                  steinvi.state, z_batch, y_batch particle_indices=particle_indices
            )
   ```

6. **Evaluation:**
   The models performance is periodically evaluated using the `evaluate_fn` attached to the `SteinVI_BNN` instance.

   In `model_training.py`:
   ```python
   if steinvi.handler._full_evaluation: current_eval_1, current_eval_2, _ = steinvi.evaluate_fn( steinvi.state, z_val, y_val, print_out=False ) # Handle early stopping if enabled
   ```

7. **Post-Training:**
   After training, you can evaluate the model on test data and visualize results.

   In `stein_vi.py`:

   ```python
   #After training is complete
   _, _, _, _, z_test, y_test = dataset
   #Evaluate on test data
   _, _, predictions_test = steinvi.evaluate_fn( steinvi.state, z_test, y_test, print_out=True )
   #Print performance metrics
   if steinvi.use_for_regression: 
      print_summary_over_particles_regression(predictions_test) 
   else: 
      print_summary_over_particles_multiclass(predictions_test) 
   ```

   Optional Visualization:
   ```python
   # Plot validation metrics over iterations
   steinvi.plot_val_metric_over_iter()

   # View misclassified examples (for classification tasks)
   steinvi.view_misclassified(z_test, y_test)
   ```

(understanding-the-main-classes)=
## Understanding the Main Classes

In this project, the primary class you will interact with is the SteinVI_BNN class, which manages the training and evaluation of Bayesian Neural Networks using Stein Variational Inference methods. Understanding this class and its associated components is crucial for configuring and executing your experiments effectively.

### `SteinVI_BNN` Class

The `SteinVI_BNN` class encapsulates the entire training process, managing particles, model configurations, and evaluation metrics. Below, we detail the key parameters and methods you need to know to set up and run your BNN.

**Class Initialization Parameters**

When creating an instance of SteinVI_BNN, you can specify various parameters to customize your training run:

- key: jax.random.PRNGKey
   - A JAX random key used for initializing particles and managing randomness.

- x_train: jax.numpy.ndarray
   - The input training data, used to initialize the neural network's parameters.
- nnet: flax.linen.Module
   - The neural network model to be optimized.
- use_for_regression: bool
   - Indicates whether the model is used for regression (True) or classification (False).
- optimizer: optax.GradientTransformation
   - The optimizer used for updating model parameters. Examples include optax.adam or optax.sgd.
- mode_training_print: str (optional, default 'none')
   - Controls the verbosity of training logs. Options are:
      - 'full': Prints detailed information during training.
      - 'reduced': Prints minimal information.
      - 'none': Disables training logs.
- mode_evaluation: str (optional, default 'full')
   - Determines the level of evaluation during training. Options are:
      - 'full': Performs comprehensive evaluation.
      - 'minimal': Collects essential evaluation metrics only.
- early_stopping: bool (optional, default False)
   - Enables early stopping based on validation metrics if set to True.
- image_data: bool (optional, default False)
   - Specifies whether the input data consists of images, affecting certain visualizations.
- batch_size: int (optional, default 0)
   -Size of data mini-batches. Set to 0 for full-batch training.
- particle_batch_size: int (optional, default 0)
   - Size of particle mini-batches. Set to 0 to update all particles simultaneously.
- num_particles: int (optional, default 10)
   - Number of particles used to approximate the posterior distribution.
- num_iterations: int (optional, default 100)
   - Total number of training iterations.
- learning_rate: float (optional, default 0.0001)
   - Learning rate used for certain algorithms. For algorithms like SVGD, the learning rate is included in the optimizer.
-  kernel_length: float (optional, default 0.005)
   - Length scale parameter for the RBF kernel used in SVGD.

```python
#Create an instance of SteinVI_BNN with custom parameters
steinvi = SteinVI_BNN(key=key, x_train=z_train, nnet=nnet_model, use_for_regression=False, optimizer=optimizer, mode_training_print='full', mode_evaluation='full', early_stopping=True, image_data=True, batch_size=300, particle_batch_size=5, num_particles=50, num_iterations=500, kernel_length=0.01 )
```

**Additional Methods and Attributes**

The SteinVI_BNN class provides several methods for model evaluation and visualization:
- `predict(z)`: Makes predictions for input data z using the trained model.
- `predict_over_particles(z)`: Returns predictions over all particles, useful for uncertainty estimation.
- `evaluate_fn(state, z_val, y_val, print_out)`: Evaluates the model on validation data.
- `plot_val_metric_over_iter()`: Plots validation metrics over training iterations.
- `plot_residuals(z_test, y_test)`: Plots residuals between predictions and true values (for regression tasks).
- `plot_location_in_relation_to_scale(z_test)`: Visualizes the relationship between prediction means and uncertainties.
- `view_misclassified(z_test, y_test)`: Displays misclassified examples and their predicted probabilities (for classification tasks).

### Configuring Training Runs

You can further customize your training runs by adjusting parameters in the Parameter and Handler classes.

**- Setting Early Stopping Criteria**

```python
#Set early stopping parameters
steinvi.parameter.set_early_stopping( warm_up_iterations=50, patience=10, min_delta=0.001 ) 
```

**- Adjusting Training Print Modes**

```python
#Set training print mode to 'reduced', 'minimal' also possible
steinvi.handler.set_training_print_mode('reduced')
```

**- Enabling Random Feature Comparison**
If you wish to compare your BNN with random feature models like Random Forests:
```python
#Enable random feature comparison
steinvi.handler.rf_comparison = True
```

A detailed example of all classes can also be found in the generated documentation under [Classes](stein_vi/Classes/index.rst).

(collected-findings)=
## Collected Findings
CURRENTLY JUST FILLET

Below is a summary of the general findings from our experiments using different Stein Variational Inference algorithms on Bayesian Neural Networks.

| **Aspect**                       | **Findings**                                                                                                                                                                                                                             |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Algorithm Performance**        | - **SVGD**: Consistently performed well across different tasks.<br>- **sSVGD**: Introduced beneficial stochasticity but required careful tuning.|
| **Number of Particles**          | - Increasing the number of particles improved posterior approximation but with diminishing returns beyond a certain point (e.g., 50 particles).<br>- More particles increased computational load linearly.                                 |
| **Batch Size**                   | - Smaller data batch sizes introduced more noise but helped in escaping local minima.<br>- Particle mini-batching helped manage memory usage when using a large number of particles.                                                      |
| **Learning Rate and Optimizers** | - Adaptive optimizers like Adam performed better than plain SGD.<br>- Learning rate schedules (e.g., exponential decay) improved training stability over long runs.                                                                     |
| **Kernel Parameters in SVGD**    | - The kernel length scale significantly impacted particle updates.<br>- Using median heuristics to initialize the kernel length scale yielded good results.<br>- Alternative kernels could be explored for better performance.            |
| **Training Dynamics**            | - Early iterations showed rapid decreases in loss, with slower improvements over time.<br>- sSVGD displayed more fluctuations due to stochasticity.|
| **Uncertainty Estimation**       | - BNNs trained with Stein VI methods provided meaningful uncertainty estimates.<br>- Particles captured different modes of the posterior, enhancing uncertainty quantification.                                                         |
| **Recommendations**              | - Use SVGD for general purposes and when resources are limited.<br>- Start with a moderate number of particles (e.g., 50).<br>- Implement early stopping to prevent overfitting. |
