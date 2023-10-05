"""Main entry point for training models."""

import logging
from enum import IntEnum
from pathlib import Path
from pprint import pformat
from random import randint

import fire
import numpy as np
import polars as pl
import pyrootutils
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from sfirah.mlp import MLPSequenceClassifier
from sfirah.transformers import EncoderTokenClassifier
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

load_dotenv()


class SpecialTokens(IntEnum):
    """Special tokens for tokernizer."""

    BOS = 0


def tokenize(example: dict) -> dict:
    """Tokenize a single example.

    Tokenizes data by converting inputs back into lists of integers; this
    allows us to leave the inputs as space-delimited strings in the CSV.
    Since we have special tokens ([BOS], etc.) we need to shift
    each token by the number of special tokens. This doesn't
    matter for the internal representations, since the element names are
    arbitrary. The output is not shifted, so the text representation of the
    input and output match [and are equal to the element index in the
    group from which they were generated].
    """
    tokenized_in = [int(t) + len(SpecialTokens) for t in str(example["input"]).split()]
    tokenized_in = [SpecialTokens.BOS.value] + tokenized_in
    tokenized_out = [
        int(t) + len(SpecialTokens) for t in str(example["target"]).split()
    ]
    tokenized_out = [SpecialTokens.BOS.value] + tokenized_out

    return {"input": tokenized_in, "target": tokenized_out}


def pad_collate(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate function for DataLoader.

    Performs channel-wise padding of the inputs and targets.
    """
    channels = samples[0].keys()
    max_lens = {}
    for channel in channels:
        max_lens[channel] = max(
            [s[channel].shape[0] if s[channel].dim() == 1 else 0 for s in samples]
        )

    for s in samples:
        for channel in channels:
            if max_lens[channel] > 0:
                s[channel] = F.pad(
                    s[channel],
                    (0, max_lens[channel] - s[channel].shape[0]),
                    value=len(SpecialTokens),
                )

    collated = {}
    for channel in channels:
        collated[channel] = torch.stack([s[channel] for s in samples])
    return collated


def get_dataset(
    group: str,
    max_len: int | None,
    k: int | None,
    train_size: float,
    data_dir: str | Path,
) -> dict:
    """Construct dataset."""
    assert train_size > 0 and train_size <= 1, "`train_size` must be in (0,1]"

    if not ((k is None) ^ (max_len is None)):
        raise ValueError("You must provide exactly one of `max_len` or `k`")

    if max_len is not None:
        assert max_len > 1, "`max_len` must be at least 2"
        data_paths = [data_dir / f"{group}={i}.csv" for i in range(2, max_len + 1)]
        if not data_paths[0].exists():
            raise FileNotFoundError(f"You must have data for {group}={2}.")
        data_paths = [p for p in data_paths if p.exists()]
        data_paths = list(set(data_paths))
        log.info("Constructing dataset from:")
        log.info("  " + "\n  ".join(map(str, data_paths)))
    else:
        assert k > 1, "`k` must be at least 2"
        data_paths = [data_dir / f"{group}={i}.csv" for i in [2, k]]
        data_paths = list(set(data_paths))
        if not data_paths[0].exists():
            raise FileNotFoundError(f"You must have data for {group}={2}.")
        log.info("Constructing dataset from:")
        log.info("  " + "\n  ".join(map(str, data_paths)))

    # We can find n_vocab just by looking at the k=2 inputs, since this is
    # guaranteed to contain all the elements in the group.
    n_vocab = (
        pl.read_csv(data_paths[0])
        .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
        .explode("input")
        .unique()
        .shape[0]
    )

    # Construct dataset
    if len(data_paths) == 1:
        dataset = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["seed"])
            .map(tokenize)
            .with_format(type="torch")
        )
        if train_size < 1:
            dataset = dataset.train_test_split(train_size=train_size)
    else:
        pair_data = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["seed"])
            .map(tokenize)
            .with_format(type="torch")
        )
        long_data = [
            (
                load_dataset("csv", data_files=str(p), split="all")
                .remove_columns(["seed"])
                .map(tokenize)
                .with_format(type="torch")
            )
            for p in data_paths[1:]
        ]

        dataset = concatenate_datasets(long_data).train_test_split(
            train_size=train_size
        )
        dataset["train"] = concatenate_datasets([dataset["train"], pair_data])

    return {
        "dataset": dataset,
        "n_vocab": n_vocab + len(SpecialTokens),
    }


def compute_metrics(data: list[(Tensor, Tensor)], prefix: str | None = None) -> dict:
    """Compute metrics."""
    values_dict = {}

    detached_data = [(d[0].cpu().detach(), d[1].cpu().detach()) for d in data]

    seq_len_max = max([d[0].shape[2] for d in detached_data])
    padded_preds = []
    padded_tgts = []
    for d in detached_data:
        pred, tgt = d
        pred_pad_size = seq_len_max - pred.shape[2]
        tgt_pad_size = seq_len_max - tgt.shape[1]

        if pred_pad_size > 0:
            padding = (pred_pad_size, 0)
            padded_pred = F.pad(pred[:, :, 1:], padding, value=1)
            padded_pred = torch.cat((pred[:, :, 0].unsqueeze(-1), padded_pred), dim=-1)
            padded_preds.append(padded_pred)
        else:
            padded_preds.append(pred)

        if tgt_pad_size > 0:
            padding = (tgt_pad_size, 0)
            padded_tgt = F.pad(tgt[:, 1:], padding, value=1)
            padded_tgt = torch.cat((tgt[:, 0].unsqueeze(-1), padded_tgt), dim=-1)
            padded_tgts.append(padded_tgt)
        else:
            padded_tgts.append(tgt)

    preds = torch.cat(padded_preds, dim=0)
    tgts = torch.cat(padded_tgts, dim=0)

    values_dict["loss"] = F.cross_entropy(preds, tgts).mean().item()
    values_dict["token_accuracy"] = (
        torch.eq(preds.argmax(dim=1), tgts).float().mean().item()
    )
    values_dict["accuracy"] = (
        torch.eq(preds.argmax(dim=1), tgts)
        .float()
        .sum(dim=1)
        .div(tgts.shape[1])
        .floor()
        .mean()
        .item()
    )

    if prefix is not None:
        values_dict = {f"{prefix}/{k}": v for k, v in values_dict.items()}

    return values_dict


def train_mlp(
    # Data parameters
    group: str,
    data_dir: Path = PROJECT_ROOT / "data",
    # Model parameters
    d_model: int = 512,
    dim_feedforward: int = 2048,
    activation: str = "relu",
    dropout: float = 0.1,
    num_layers: int = 1,
    weight_sharing: bool = False,
    weight_scale: float = 1.0,
    layer_norm_eps: float = 1e-05,
    bias: bool = True,
    # Training parameters
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    op_eps: float = 1e-8,
    weight_decay: float = 0.01,
    # Misc
    log_level: str = "INFO",
    seed: int = randint(0, 2**32 - 1),
    project_name: str = "word_problems_mlp",
    logging: bool = True,
):
    """Train MLP model."""
    set_seed(seed)

    accelerator = Accelerator(log_with="wandb") if logging else Accelerator()
    log.setLevel(log_level)

    # Load dataset
    datadict = get_dataset(group, max_len=None, k=2, train_size=1.0, data_dir=data_dir)
    dataset = datadict["dataset"]
    n_vocab = datadict["n_vocab"]
    log.info(f"Dataset: {dataset}")

    # Set up logger
    project_hps = {
        "group": group,
        "d_model": d_model,
        "dim_feedforward": dim_feedforward,
        "activation": activation,
        "dropout": dropout,
        "num_layers": num_layers,
        "weight_sharing": weight_sharing,
        "weight_scale": weight_scale,
        "layer_norm_eps": layer_norm_eps,
        "bias": bias,
        "n_vocab": n_vocab,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": op_eps,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    log.info(f"Config: {pformat(project_hps)}")

    accelerator.init_trackers(
        project_name,
        config=project_hps,
    )

    model = MLPSequenceClassifier(
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        activation=activation,
        dropout=dropout,
        num_layers=num_layers,
        weight_sharing=weight_sharing,
        weight_scale=weight_scale,
        layer_norm_eps=layer_norm_eps,
        bias=bias,
        n_vocab=n_vocab,
        seq_len=3,
    )
    log.info(f"Model: {model}")
    log.info(f"Accelerator state: {accelerator.state}")

    device = accelerator.device
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=op_eps,
        weight_decay=weight_decay,
    )
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=pad_collate,
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    global_step = 0
    for epoch in (n_bar := tqdm(range(epochs), desc="Epochs", position=0, leave=False)):
        model.train()
        train_loss = []
        train_metric_data = []
        for batch in (
            t_bar := tqdm(train_dataloader, desc="Train", position=1, leave=False)
        ):
            global_step += 1
            optimizer.zero_grad()

            source = batch["input"]
            target = batch["target"]
            output = model(source)

            loss = F.cross_entropy(output, target)
            train_loss.append(loss.item())

            predictions, references = accelerator.gather_for_metrics((output, target))

            train_metric_data.append((predictions, references))

            accelerator.backward(loss)
            optimizer.step()

            t_bar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_metrics = compute_metrics(train_metric_data, prefix="train")
        train_metrics["train/loss"] = np.mean(train_loss)
        n_bar.set_postfix(
            {
                "loss": f"{train_metrics['train/loss']:.5f}",
                "acc": f"{train_metrics['train/accuracy']:.5f}",
            }
        )
        accelerator.log(train_metrics, step=global_step)
        accelerator.log({"epoch": epoch}, step=global_step)

    log.info(train_metrics)
    accelerator.end_training()


def train(
    # Data parameters
    group: str,
    data_dir: Path = PROJECT_ROOT / "data",
    max_len: int | None = None,
    k: int | None = None,
    train_split: float = 0.8,
    # Model parameters
    d_model: int = 512,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str = "relu",
    layer_norm_eps: float = 1e-5,
    norm_first: bool = False,
    num_layers: int = 1,
    weight_sharing: bool = False,
    weight_scale: float = 1.0,
    bias: bool = True,
    # Training parameters
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    op_eps: float = 1e-8,
    weight_decay: float = 0.01,
    compile: bool = False,
    # Misc
    log_level: str = "INFO",
    seed: int = randint(0, 2**32 - 1),
    project_name: str = "word_problems",
    logging: bool = True,
):
    """Train transformer model."""
    set_seed(seed)

    accelerator = Accelerator(log_with="wandb") if logging else Accelerator()
    log.setLevel(log_level)

    # Load dataset
    datadict = get_dataset(group, max_len, k, train_split, data_dir)
    dataset = datadict["dataset"]
    n_vocab = datadict["n_vocab"]
    log.info(f"Dataset: {dataset}")

    # Set up logger
    project_hps = {
        "max_len": max_len,
        "group": group,
        "d_model": d_model,
        "compile": compile,
        "nhead": nhead,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "activation": activation,
        "layer_norm_eps": layer_norm_eps,
        "norm_first": norm_first,
        "num_layers": num_layers,
        "weight_sharing": weight_sharing,
        "weight_scale": weight_scale,
        "bias": bias,
        "n_vocab": n_vocab,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": op_eps,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    log.info(f"Config: {pformat(project_hps)}")

    accelerator.init_trackers(
        project_name,
        config=project_hps,
    )

    # Construct model
    model = EncoderTokenClassifier(
        d_model=d_model,
        n_heads=nhead,
        d_ff=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
        n_layers=num_layers,
        weight_sharing=weight_sharing,
        weight_scale=weight_scale,
        n_vocab=n_vocab,
        batch_first=True,
        bias=bias,
    )

    if compile:
        log.info("Compiling model...")
        # TODO: This may not work on CUDA. It doesn't seem like
        # this should be necessary since
        # # https://github.com/pytorch/pytorch/pull/96980 was merged, but
        # not specifying the backend causes an error.
        model = torch.compile(model, backend="aot_eager")
        log.info("Model compiled!")

    log.info(f"Model: {model}")
    log.info(f"Accelerator state: {accelerator.state}")

    device = accelerator.device

    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=op_eps,
        weight_decay=weight_decay,
    )
    if train_split < 1:
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=pad_collate,
        )
        eval_dataloader = DataLoader(
            dataset["test"],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=pad_collate,
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=pad_collate,
        )
        eval_dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=pad_collate,
        )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    global_step = 0
    for epoch in (n_bar := tqdm(range(epochs), desc="Epochs", position=0, leave=False)):
        model.train()
        train_data = []
        for batch in (
            t_bar := tqdm(train_dataloader, desc="Train", position=1, leave=False)
        ):
            global_step += 1
            optimizer.zero_grad()

            source = batch["input"]
            target = batch["target"]
            output = model(source)

            loss = F.cross_entropy(output, target)

            predictions, references = accelerator.gather_for_metrics((output, target))

            train_data.append((predictions, references))

            if np.random.random_sample() < 0.01:
                print(f"preds: {predictions.argmax(dim=1)}")
                print(f"trgts: {references}")

            accelerator.backward(loss)
            optimizer.step()

            t_bar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_metrics = compute_metrics(train_data, prefix="train")
        accelerator.log(train_metrics, step=global_step)

        model.eval()
        eval_data = []
        for batch in tqdm(eval_dataloader, desc="Eval", position=1, leave=False):
            source = batch["input"]
            target = batch["target"]
            with torch.no_grad():
                output = model(source)

            predictions, references = accelerator.gather_for_metrics((output, target))

            eval_data.append((predictions, references))

        eval_metrics = compute_metrics(eval_data, prefix="val")
        accelerator.log(eval_metrics, step=global_step)
        accelerator.log({"epoch": epoch}, step=global_step)

        n_bar.set_postfix({"val/acc": eval_metrics["val/accuracy"]})

    log.info(eval_metrics)
    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire()
