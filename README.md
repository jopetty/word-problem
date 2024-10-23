# Word Problems

Data and code to evaluate how deep transformer models need to be to learn group multiplication.

## Installation

Dependencies are specified in `pyproject.toml` and can be installed with any Python package manger (e.g., `pip` or `poetry`). For convenience, we also provide a Conda `environment.yml` file which wraps around `pyproject.toml`. To create the environment and install all dependencies, run

```bash
conda env create && conda activate wp
```

## Use

### Training Transformer Models

To train a model, run `python src/main.py train`. Command-line arguments to configure the training run are the arguments to the `train` function, which is the most accurate documentation for what to do. Some important ones:

- `--group`: the name of the group to train on.
- `--k`: the length of the sequences to train on. Must be greater than 1.
- `--num_layers`: how many layers in the transformer encoder.
- `--epochs`: the number of epochs to train for.

Training on sequences of length `k` means that all `group=k` sequences will be split between the train and test sets. Additionally, all sequences of length 2 <= `m` < `k` will be included in the training set; to _only_ train on sequences of length 2 and `k`, pass the `--strict_len` flag.

```bash
# Trains a model on all sequences of length 2, 3, 4 on data from S5
python src/main.py train --group S5 --k 4

# Trains a model only on sequences of length 2 and 4 on data from S5
python src/main.py train --group S5 --k 4 --strict_len
```

The combination of `group` and `k` determines which data files to use. Data files are stored by default in the `data/` directory and have the name `group=k.csv`.

### Training MLP Baselines

As a sanity check for what a single-layer should be able to compute, we can train a MLP to learn binary multiplication. We don't train with a train/test split since we are only concerned with whether or not a single layer can learn the function, not how well it generalizes. The only required argument is `--group`.

```bash
# Train a single-layer MLP on binary multiplication in Z60
python src/main.py train_mlp --group Z60
```

### Generating Data

Data files are generated by calling `python src/generate_data.py` with the following arguments:

- `--group`: A string representing the group to use. Supported groups are of the form `G#`, where `G` is one of `S` (symmetric), `A` (alternating), or `Z` (cyclic), and `#` is an integer. You can also use the direct products of any of these groups by separating each with a `_x_`. So `--group=S5` generates data using elements of `S5` and `--group=A5_x_Z9` generates data using elements of `A5 x Z9`.
- `--k`: The length of the sequences to generate. Each sequence consists of `k` elements multiplied together.
- `--samples`: The number of examples to generate. If this is left blank, it will generate the maximum number of distinct sequences possible for the given group and sequence length, equal to `#(G)^k`. If this is set to be an integer, it will generate `min(samples, #(G)^k)` examples; note that we cap the number of examples to ensure that any partitions of the generate datasets are guaranteed to be sequence-wise disjoint. If `samples` is less than `#(G)^k`, examples will be generated randomly without replacement.
- `--data_dir`: The directory to save the data to. If this is left blank, it will save to a `data/` directory in the project root.
- `--seed`: The random seed to use for generating the data.
- `--overwrite`: Whether to overwrite an existing data file for the given values for `group` and `k`.

```bash
# Generate 100k 5-element sequences & their reductions from S5
python src/generate_data.py --group S5 --k 5 --samples 100000
```

Data files are named `group=k.csv`.

```csv
length,input,target
```

where `length` is equal to `k`, `input` is a series of space-separated integers corresponding to the element index of the group as defined in the `abstract_algebra` object, and `target` is the element the sequence of input elements multiplies to (again, as the element index of the group object).

By default, we include data files for `Z60`, `A4_x_Z5`, and `A5` for sequence lengths 2 through 10, subsampled to a maximum of 5,000,000 examples per `k`.

### Logging

Run data is logged to Weights&Biases; add `WANDB_API_KEY=##################` to a `.env` file in the root directory of the project to automatically configure the W&B logger. If you don't want to log runs to W&B, pass the `--nologging` flag.


## Setting up on SLURM

Refer to [the NYU HPC/Singularity Guide](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda) for more thorough instructions.

1. Clone this repository to scratch (e.g., `/scratch/NetID/word-problem`). `cd` into this directory.
2. Copy the default singularity overlay:

```bash
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz
```

3. Launch the Singularity container in read-write mode:
```bash
singularity exec --overlay overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
```

4. Install Miniconda:
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
# rm Miniconda3-latest-Linux-x86_64.sh # if you don't need this file any longer
```

5. Create the `/ext3/env.sh` activation script:

```bash
vim /ext3/env.sh
```

Enter and save the following:

```bash
#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
```

5. Install the Conda environment:
```bash
source /ext3/env.sh
conda env create -f environment.yml
```

This should install all dependencies inside the `wp` environment.

NOTE: I'm not 100% certain if this will install the appropriate CUDA libraries if you are not running this from a GPU node. To make sure this works, you can `srun` a job on a GPU, launch the Singularity container, `source /ext3/env.sh`, and then load up a Python REPL, `import torch`, and confirm GPU access by running `torch.cuda.is_available()`.

Once you've done this, you should be able to run, e.g., `bash run-ids4.sh` to launch all the training jobs.

## Will's Finetuning Experiments

I removed the following line from pyproject.toml since we don't need SSM dependencies:
```
    "sfirah [ssm] @ git+https://github.com/jopetty/sfirah ; sys_platform != 'darwin'",
```

To generate data:
```shell
ROOT="/net/nfs.cirrascale/allennlp/willm/log-depth"
N_TRAIN=1000000
N_VAL=1000
KS=("2" "4" "8" "16" "32" "64" "128")

for k in "${KS[@]}"; do
    mkdir $ROOT/data/$k
    python src/generate_data.py A5 --k $k --data-path $ROOT/data/$k/train.csv --samples $N_TRAIN
    python src/generate_data.py A5 --k $k --data-path $ROOT/data/$k/val.csv --samples $N_VAL
done
```

Switching to using Gantry to run finetuning:

```shell
./finetune.sh
```