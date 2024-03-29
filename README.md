# A list of topics I want to learn over winter 2022/2023

## General Goal

I wanted to learn rust and machine learning..so I thought why not do ML in rust. The notebooks, while in python, have rust bindings to the core ML/DL algorithms. <br/>

I have also written my own cuda matrix library, which I have painstakingly optimized to use batched operations, chunked memory management, asynchronous memory transfers, mempools/async mem management and more. <br/>
Why not use cublas you ask?

- It was a good learning exercise, I don't pretend it is faster than existing frameworks
- CUBLAS is there, just commented out (I used it for performance comparison and I am usually within a factor of 2-3x)

#### NO ML CODE WAS WRITTEN IN PYTHON! <br/>

#### Python is just used for preprocessing and loading of data, visualization and verification of results.<br/>

That means all the ml algorithms are written using good old if/else, for loops etc. No significant help from libraries.<br/><br/>
The following are the only libraries used for actual ML logic:

- Rust's itertools for iterating over data more easily
- Rust's statsrs and Python's numpy for random number generation
- Rust's Image lib to decode thousands of images of varying formats into a raw float array efficiently

## Structure

PYO3 is used to create bindings from python to the RUST binaries.
The python code is located under notebooks.
The rust core ML functions are located under src.
Any data is located in the data folder.

### Goals

| Algorithm                          | Status             |
| ---------------------------------- | ------------------ |
| K-Means                            | :white_check_mark: |
| K-Nearest Neighbors                | :white_check_mark: |
| Naive Bayes                        | :white_check_mark: |
| Decision Trees/Random Forest       | :white_check_mark: |
| Regression Tree                    | :white_check_mark: |
| Gradient Descent                   | :white_check_mark: |
| ADA Boost                          | :white_check_mark: |
| Gradient Boost                     | :white_check_mark: |
| XGBoost                            | :white_check_mark: |
| Neural Network w/t backpropogation | :white_check_mark: |
| Convolutional Neural Networks      | :white_check_mark: |
| Recurrent Neural Networks          | :x:                |
| Generative Adversarial Networks    | :x:                |
| Large Language Models              | :x:                |
| CUDA Acceleration w/t Rust FFI     | :white_check_mark: |

## Setup Instructions

1. Install python3
2. Install Rust
3. Install nvidia cuda toolkit
4. Create .venv folder `python -m venv .venv`
5. Activate python virtual environment (platform dependent)<br/>
   Windows: `.\.venv\Scripts\activate.bat`<br/>
   Mac/Linux: `source ./.venv/bin/activate`
6. Install dependencies from requirements.txt<br/>
   `pip install -r requirements_{platform}.txt`
7. Compile Rust Code using `maturin develop` or `maturin develop --release`
8. Open notebooks in jupyter notebook/jupyter lab/vscode etc..

## Running Tests

Make sure to run tests using `cargo test -- --test-threads=1`.
Running the tests in parallel may fail because the cuda matrix library is not thread safe (yet).

## Credits

MNIST Handwritten digit database available from https://yann.lecun.com/exdb/mnist/ <br/>
All Other datasets are publically available from University of California Irvine here: https://archive.ics.uci.edu/ml/index.php
