# MA-223 Exam Rust ML
This repo does machine learning using open-source tensor and deep learning framework:  [burn](https://github.com/tracel-ai/burn).
The reason for choosing burn as a backend framework over something like tensorflow in python. Is that burn automatically chooses the support GPU or CPU backend based on hardware.

## Installing cargo and rust
Install `cargo` and `rust` by installing `rustup`. See https://doc.rust-lang.org/cargo/getting-started/installation.html.

## Setting up experiments
Experiments to run are generated based on the `config/config.json` and `config/hyperparameters.json` files. It runs every combination of set variables. 

For instance if you have set:
```json
{
"max_epochs": [25],
"dropout": [0.4],
"learning_rate": [0.001, 0.05],
"activation": ["gelu", "linear"]
}
```
Experiments, `gelu@0.001LR`, `gelu@0.05LR`, `linear@0.001LR`, and `linear@0.05LR` will run.
The `hyperparameters.json` file is for experimenting with hyperparamters. And `config.json` file is for more general experiment setup. Such as the name.

## Running experiments
Once the config is set up. You are ready to run the experiments. Run the experiments with `cargo run --release` for better performance. Results will be organized into a results/experiment_name folder. 

# Dataset
[IAM Handwriting Top50 dataset](https://www.kaggle.com/datasets/tejasreddy/iam-handwriting-top50/data) from user TEJASREDDY. Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
