# A list of topics I want to learn over winter 2022/2023

## General Goal

I wanted to learn rust and machine learning..so I thought why not do ML in rust. The notebooks, while in python, have rust bindings to the core ML/DL algorithms.

#### NO ML CODE WAS WRITTEN IN PYTHON! <br/>

#### Python is just used for preprocessing and loading of data, visualization and verification of results.<br/>

That means all the ml algorithms are written using good old if/else, for loops etc. No significant help from libraries (unless you count random number generation and Rust's amazing itertools).

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
| Convolutional Neural Networks      | :x:                |
| Recurrent Neural Networks          | :x:                |
| Generative Adversarial Networks    | :x:                |
| CUDA Acceleration w/t pybind11     | :x:                |

## Setup Instructions

1. Install python3
2. Install Rust
3. Activate python virtual environment (platform dependent)<br/>
   Windows: `.\.venv\Scripts\activate.bat`<br/>
   Mac/Linux: `source ./.venv/bin/activate`
4. Install dependencies from requirements.txt<br/>
   `pip install -r requirements.txt`
5. Build cuda kernels<br/>
   - `cd matrix_lib`
   - compile_cuda_kernels.bat (Windows)<br/>
     chmod +x compile_cuda_kernels.sh && ./compile_cuda_kernels.sh (Linux)
   - `cd ..`
6. Compile Rust Code using `maturin develop` or `maturin develop --release`
7. Open notebooks in jupyter notebook/jupyter lab/vscode etc..

## Credits

MNIST Handwritten digit database available from https://yann.lecun.com/exdb/mnist/ <br/>
All Other datasets are publically available from University of California Irvine here: https://archive.ics.uci.edu/ml/index.php
