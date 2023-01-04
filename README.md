# A list of topics I want to learn over winter 2022/2023

## General Goal

Learn rust and machine learning. The notebooks, while in python, have rust bindings to the core ML/DL algorithms.

### Goals

1. K-Means
2. K-Nearest Neighbors
3. Naive Bayes
4. Decision Trees, Regression Tree, Random Forest
5. Basic Neural Network
6. With backpropogation
7. Convolutional Neural Networks
8. Recurrent Neural Networks
9. Generative Adversarial Networks

## Setup Instructions

1. Install python3
2. Install Rust
3. Activate python virtual environment (platform dependent)
4. Install dependencies from requirements.txt
5. Install jupyter notebook/jupyter lab
6. Open notebooks in jupyter notebook/jupyter lab

\*\* Note

K-Means-Clustering.ipynb uses evcxr which is purely in rust. The rest of the notebook is in python with bindings to core rust code.

The switch is due to how slow dependencies installed in evcxr everytime the notebook was opened (even with caching enabled). And the general tooling is better with python.

EVCXR is a very cool project though!
