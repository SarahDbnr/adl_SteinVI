# Bayesian Neural Networks through Stein VI in JAX documentation

## Collaborative Work
This repository is a collaborative work by **Sarah Deubner**, **Kilian Runnwerth**, and **Luke-Liam Bergmeier**.

## Introduction
This repository employs a Stein Variational Inference (Stein VI) approach to apply Bayesian Inference via Variational Gradient Descent (VGD) for optimizing Bayesian Neural Networks (BNNs). The work is based on the ideas presented in the following paper: 
[Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471).

```{toctree}
:maxdepth: 3
:caption: Documentation:

get_started
src/BNN_Model
src/Parameter_Class
src/run_SVGD
src/get_posteriori
src/svgd
src/data_handling
src/datasets_info
src/plots_validation_metrics
src/print_evaluation
src/regression_toy_example
src/run_SVGD_and_return_evaluation_values
src/validation_and_evaluation