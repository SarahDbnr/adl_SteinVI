# Introduction
This repository uses a Stein VI approach to apply Bayesian Inference via Variational Gradient Descent to 
optimize Bayesian Neural Networks.
Based on the ideas of this paper: https://urldefense.com/v3/__https://arxiv.org/abs/1608.04471__;!!JTSHVUr6R1OOzg!INusezDlEUF5xiznfrs_AlkUNTt5jFRpBSsQILS5AT0ewEB5KRz0D8O-6uQqOJzUj61HMRWk471962hhKzJk2EY$ <br><br>
This repository includes the following topics:
1. Examples for bayesian neural networks using the Blackjax library where svgd are already implemented

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