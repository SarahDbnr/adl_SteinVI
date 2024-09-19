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

stein_vi

stein_vi/Classes/Handler_Class
stein_vi/Classes/Parameter_Class
stein_vi/Classes/SteinVI_BNN

stein_vi/algorithm/get_posteriori
stein_vi/algorithm/model_training
stein_vi/algorithm/random_forest
stein_vi/algorithm/svgd

stein_vi/metrics/plots_validation_metrics
stein_vi/metrics/validation_and_evaluation
stein_vi/metrics/view_misclassified_images

stein_vi/parameter_search/print_evaluation
stein_vi/parameter_search/run_SVGD_and_return_evaluation_values

