# Collaborative Work
This repository is a collaborative work by **Sarah Deubner**, **Kilian Runnwerth**, and **Luke-Liam Bergmeier**.

# Introduction
This repository employs a Stein Variational Inference (Stein VI) approach to apply Bayesian Inference via Variational Gradient Descent (VGD) for optimizing Bayesian Neural Networks (BNNs). The work is based on the ideas presented in the following paper: 
[Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471).

### Included Topics
This repository includes the following:
1. Examples of Bayesian Neural Networks using the Blackjax library, where SVGD is already implemented.
2. Implementation of advanced SVGD algorithms such as sSVGD, SVN, sSVN. Integrated in the blackjax package. # TODO

# Installation
1. Download and install python 3.11
2. Create a new environment using `pipenv install --dev`
3. Create new shell using `pipenv shell`

# Getting started

## To run BNN Toy Example 
run main in file `BNN_Example_clean_version/run_SVGD`\
choose dataset by running the according function at the bottom of the file, for example for the dataset MNIST:\
`if __name__ == "__main__":
    run_MNIST()`

## Test 
To run all tests run `pytest tests` !! # TODO: test is file with tests we still need that
# Parameters for Datasets
## MNIST
| **NETWORK_STRUCTURE**            | **NUM_Particals**                                | **Optimizer**  | **Learning Rate**                       | **Batch size** | **Decay Rate** | **Test MSE** |
|-----------------------|-----|-----------------------|-------|-----|------|-------------------|
| (200, 75, 40) | 12 | exponential-adam |  0.05 | 300 | 0.95 |  0.9758999943733215 |


