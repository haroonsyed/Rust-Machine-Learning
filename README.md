# A list of topics I want to learn over winter 2022/2023

## General Goal

Learn rust and machine learning. The notebooks, while in python, have rust bindings to the core ML/DL algorithms.

## Structure

PYO3 is used to create bindings from python to the RUST binaries.
The python code is located under notebooks.
The rust core ML functions are located under src.
Any data is located in the data folder.

### Goals

1. K-Means
2. K-Nearest Neighbors
3. Naive Bayes
4. Decision Trees/Random Forest
5. Regression Tree
6. Gradient Descent
7. ADA Boost
8. Gradient Boost
9. XGBoost (Regression only, no parallelism/histogram optimizations)
10. Basic Neural Network
11. With backpropogation
12. Convolutional Neural Networks
13. Recurrent Neural Networks
14. Generative Adversarial Networks
15. Acceleration on CUDA with pybind11 bindings

## Setup Instructions

1. Install python3
2. Install Rust
3. Activate python virtual environment (platform dependent)
4. Install dependencies from requirements.txt
5. Compile Rust Code using `maturin develop` or `maturin develop --release`
6. Install jupyter notebook/jupyter lab
7. Open notebooks in jupyter notebook/jupyter lab

## Credits

All datasets are publically available from University of California Irvine here: https://archive.ics.uci.edu/ml/index.php
