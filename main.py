import logging
from enum import IntEnum, auto
from pathlib import Path
from pprint import pformat
from random import randint

import fire
import numpy as np
import polars as pl
import pyrootutils
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from evaluate import load as load_metric
from torch import Tensor, nn, optim
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
    CLS = 0
    PAD = auto()


class AvgPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class IndexPool(nn.Module):
    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x: Tensor) -> Tensor:
        return x.select(dim=self.dim, index=self.index)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, index={self.index}"


def get_activation(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "relu6":
        return nn.ReLU6()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class MLPModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        layers: int,
        n_vocab: int,
        layer_norm_eps: float = 1e-05,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab + len(SpecialTokens), d_model)
        ff_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.LayerNorm(d_model, layer_norm_eps),
        )
        self.ff = nn.ModuleList([ff_layer for _ in range(layers)])
        self.pool = IndexPool(dim=1, index=0)
        self.classifier = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        x = self.embedding(x)
        for ff_layer in self.ff:
            x = ff_layer(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits


class EncoderModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        norm_first: bool,
        num_layers: int,
        weight_sharing: bool,
        n_vocab: int,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )
        self.weight_sharing = weight_sharing
        self.num_layers = num_layers
        self.embedding = nn.Embedding(n_vocab + len(SpecialTokens), d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        if weight_sharing:
            if num_layers == 1:
                print("Weight sharing has no effect for single-layer models.")
            for i in range(1, num_layers):
                self.encoder.layers[i].self_attn.in_proj_weight = self.encoder.layers[
                    0
                ].self_attn.in_proj_weight
                self.encoder.layers[i].self_attn.in_proj_bias = self.encoder.layers[
                    0
                ].self_attn.in_proj_bias
                self.encoder.layers[i].self_attn.out_proj.weight = self.encoder.layers[
                    0
                ].self_attn.out_proj.weight
                self.encoder.layers[i].self_attn.out_proj.bias = self.encoder.layers[
                    0
                ].self_attn.out_proj.bias
                self.encoder.layers[i].linear1.weight = self.encoder.layers[
                    0
                ].linear1.weight
                self.encoder.layers[i].linear1.bias = self.encoder.layers[
                    0
                ].linear1.bias
                self.encoder.layers[i].linear2.weight = self.encoder.layers[
                    0
                ].linear2.weight
                self.encoder.layers[i].linear2.bias = self.encoder.layers[
                    0
                ].linear2.bias
                self.encoder.layers[i].norm1.weight = self.encoder.layers[
                    0
                ].norm1.weight
                self.encoder.layers[i].norm1.bias = self.encoder.layers[0].norm1.bias
                self.encoder.layers[i].norm2.weight = self.encoder.layers[
                    0
                ].norm2.weight
                self.encoder.layers[i].norm2.bias = self.encoder.layers[0].norm2.bias

        self.pool = IndexPool(dim=1, index=0)
        self.classifier = nn.Linear(d_model, n_vocab)

    def forward(self, x):

        if self.weight_sharing:
            assert torch.equal(
                self.encoder.layers[self.num_layers - 1].linear1.weight,
                self.encoder.layers[0].linear1.weight,
            ), "Weights not tied!"

        x = self.pos_enc(self.embedding(x))
        x = self.encoder(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits


def tokenize(example: dict) -> dict:
    # "Tokenize" data by converting inputs back into lists of integers;
    # allows us to leave the inputs as space-delimited strings in the CSV.
    # Since we have a [CLS] token, each token is shifted by 1. This doesn't
    # matter for the internal representations, since the element names are
    # arbitrary. The output is not shifted, the text representation of the
    # input and output match [and are equal to the element index in the
    # group from which they were generated].
    tokenized = [int(t) + len(SpecialTokens) for t in str(example["input"]).split()]
    tokenized = [SpecialTokens.CLS.value] + tokenized

    return {"input": tokenized, "target": int(example["target"])}


def pad_collate(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:

    # Perform channel-wise padding
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
                    value=SpecialTokens.PAD.value,
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
            raise FileNotFoundError(f"You must have data for {group}={1}.")
        log.info("Constructing dataset from:")
        log.info("  " + "\n  ".join(map(str, data_paths)))

    # We can find n_vocab just by looking at the k=2 data, since this is
    # guaranteed to contain all the elements in the group.
    n_vocab = (
        pl.read_csv(data_paths[0])
        .with_columns(
            pl.concat_str([pl.col("input"), pl.col("target")], separator=" ").alias(
                "merged"
            )
        )
        .select(pl.col("merged").map_batches(lambda x: x.str.split(" ")))
        .explode("merged")
        .unique()
        .shape[0]
    )

    # Construct dataset
    if len(data_paths) == 1:
        dataset = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["length"])
            .map(tokenize)
            .with_format(type="torch")
        )
        if train_size < 1:
            dataset = dataset.train_test_split(train_size=train_size)
    else:
        pair_data = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["length"])
            .map(tokenize)
            .with_format(type="torch")
        )
        long_data = [
            (
                load_dataset("csv", data_files=str(p), split="all")
                .remove_columns(["length"])
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
        "n_vocab": n_vocab,
    }


def compute_metrics(metrics: list, prefix: str | None = None) -> dict:
    values_dict = {}
    for metric in metrics:
        simple_name = metric.name.split("/")[-1]
        if simple_name in ["accuracy", "confusion_matrix"]:
            values_dict[simple_name] = metric.compute()[metric.name]
        elif simple_name in ["precision", "recall"]:
            values_dict[simple_name] = metric.compute(
                average="weighted", zero_division=0
            )[metric.name]
        elif simple_name == "f1":
            values_dict[simple_name] = metric.compute(average="weighted")[metric.name]

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
    layers: int = 1,
    # Training parameters
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    op_eps: float = 1e-8,
    weight_decay: float = 0.01,
    # Misc
    log_level: str = "INFO",
    seed: int = randint(0, 1_000_000),
    project_name: str = "word_problems_mlp",
    logging: bool = True,
):

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
        "layers": layers,
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

    model = MLPModel(
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        activation=activation,
        dropout=dropout,
        layers=layers,
        n_vocab=n_vocab,
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

    metrics = [load_metric("accuracy")]

    global_step = 0
    for epoch in (n_bar := tqdm(range(epochs), desc="Epochs", position=0, leave=False)):
        model.train()
        train_loss = []
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

            # log.debug(f"Predictions: {predictions.argmax(dim=-1)}")
            # log.debug(f"References: {references}")

            for metric in metrics:
                metric.add_batch(
                    predictions=predictions.argmax(dim=-1), references=references
                )
            accelerator.backward(loss)
            optimizer.step()

            t_bar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_metrics = compute_metrics(metrics, prefix="train")
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
    num_layers: int = 6,
    weight_sharing: bool = False,
    # Training parameters
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    op_eps: float = 1e-8,
    weight_decay: float = 0.01,
    # Misc
    log_level: str = "INFO",
    seed: int = randint(0, 2**32 - 1),
    project_name: str = "word_problems",
    logging: bool = True,
):

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
        "nhead": nhead,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "activation": activation,
        "layer_norm_eps": layer_norm_eps,
        "norm_first": norm_first,
        "num_layers": num_layers,
        "weight_sharing": weight_sharing,
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
    model = EncoderModel(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
        num_layers=num_layers,
        weight_sharing=weight_sharing,
        n_vocab=n_vocab,
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

    # Construct metrics
    metrics = [
        load_metric("accuracy"),
        load_metric("precision"),
        load_metric("recall"),
        load_metric("f1"),
        load_metric("BucketHeadP65/confusion_matrix"),
    ]

    global_step = 0
    for epoch in (n_bar := tqdm(range(epochs), desc="Epochs", position=0, leave=False)):
        model.train()
        train_loss = []
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

            for metric in metrics:
                metric.add_batch(
                    predictions=predictions.argmax(dim=-1), references=references
                )
            accelerator.backward(loss)
            optimizer.step()

            t_bar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_metrics = compute_metrics(metrics, prefix="train")
        train_metrics["train/loss"] = np.mean(train_loss)
        accelerator.log(train_metrics, step=global_step)

        model.eval()
        for batch in (
            e_bar := tqdm(eval_dataloader, desc="Eval", position=2, leave=False)
        ):
            source = batch["input"]
            target = batch["target"]
            with torch.no_grad():
                output = model(source)
            predictions, references = accelerator.gather_for_metrics((output, target))
            for metric in metrics:
                metric.add_batch(
                    predictions=predictions.argmax(dim=-1), references=references
                )

        eval_metrics = compute_metrics(metrics, prefix="val")
        accelerator.log(eval_metrics, step=global_step)
        accelerator.log({"epoch": epoch}, step=global_step)

        n_bar.set_postfix({"val/acc": eval_metrics["val/accuracy"]})

    log.info(eval_metrics)
    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire()
