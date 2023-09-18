# Word Problems

Data and code to evaluate how deep transformer models need to be to learn group multiplication.

## Installation

If you only want to use the built-in data, you can just clone the repo and run `conda env create` from the root directory. If you want to generate new data, you'll need to install the [`abstract_algebra` package](https://abstract-algebra.readthedocs.io/en/latest/index.html). Note that I think the package is a bit misconfigured, so installing with pip via a git link doesn't work; you'll need to install it directly once you've created the virtual environment.

## Use

To train a model, run `python depth_test.py`. Command-line arguments to configure the training run are the arguments to the `main` function, which is the most accurate documentation for what to do. Some important ones:

- `data`: the name of the file in the `data/` directory, without the `.csv` extension.
- `num_layers`: how many layers in the transformer encoder.
- `epochs`: the number of epochs to train for.

### Logging

Run data is logged to Weights&Biases; add `WANDB_API_KEY=##################` to a `.env` file in the root directory of the project to automatically configure the W&B logger
