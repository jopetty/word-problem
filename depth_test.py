from pathlib import Path
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
from datasets import load_dataset
from dotenv import load_dotenv
from evaluate import load as load_metric
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

log = get_logger(__name__, log_level="INFO")

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

load_dotenv()


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


class EncoderModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        batch_first: bool,
        norm_first: bool,
        num_layers: int,
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
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.embedding = nn.Embedding(n_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))
        x = self.encoder(x)
        x = x[:, 0, :]
        logits = self.classifier(x)
        return logits


def tokenize(example: dict) -> dict:
    # "Tokenize" data by converting inputs back into lists of integers;
    #  allows us to leave the inputs as space-delimited strings in the CSV
    specials = ["[CLS]"]
    tokenized = [int(t) for t in str(example["input"]).split()]
    tokenized = [specials.index("[CLS]")] + tokenized
    return {"input": tokenized, "target": int(example["target"])}


def compute_metrics(metrics: list, prefix: str | None = None) -> dict:
    values_dict = {}
    for metric in metrics:
        if metric.name == "accuracy":
            values_dict[metric.name] = metric.compute()["accuracy"]
        elif metric.name in ["precision", "recall"]:
            values_dict[metric.name] = metric.compute(
                average="weighted", zero_division=0
            )[metric.name]
        elif metric.name == "f1":
            values_dict[metric.name] = metric.compute(average="weighted")[metric.name]

    if prefix is not None:
        values_dict = {f"{prefix}/{k}": v for k, v in values_dict.items()}

    return values_dict


def main(
    data: str = "S5=1",
    data_dir: str | Path = PROJECT_ROOT / "data",
    mode: str = "train",
    # Transformer parameters
    d_model: int = 512,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str = "relu",
    layer_norm_eps: float = 1e-5,
    batch_first: bool = True,
    norm_first: bool = False,
    num_layers: int = 6,
    # Training parameters
    train_split: float = 0.8,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    # Misc
    log_level: str = "INFO",
    seed: int = randint(0, 1_000_000),
    project_name: str = "word_problems",
    use_logger: bool = True,
):

    set_seed(seed)

    accelerator = Accelerator(log_with="wandb") if use_logger else Accelerator()
    log.setLevel(log_level)

    assert mode in ["train", "test"], "mode must be either 'train' or 'test'"

    # Load dataset
    assert train_split > 0 and train_split < 1, "train_split must be between 0 and 1"
    data_path = str(data_dir / f"{data}.csv")
    # dataset = load_dataset("csv", data_files=data_path, split="all")
    dataset = (
        load_dataset("csv", data_files=data_path, split="all")
        .remove_columns(["length"])
        .map(tokenize)
        .with_format(type="torch")
        .train_test_split(train_size=train_split)
    )
    # dataset = dataset.remove_columns(["length"])
    # dataset = dataset.map(tokenize)
    # dataset = dataset.with_format(type="torch")
    # dataset = dataset.train_test_split(train_size=train_split)

    log.info("Dataset: ", dataset)
    print(dataset)

    # Calculate n_vocab dynamically by counting unique tokens in dataset
    n_vocab = (
        pl.read_csv(data_path)
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

    # Set up logger
    project_hps = {
        "sequence_length": int(data.split("=")[1]),
        "dataset": data.split("=")[0],
        "d_model": d_model,
        "nhead": nhead,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "activation": activation,
        "layer_norm_eps": layer_norm_eps,
        "batch_first": batch_first,
        "norm_first": norm_first,
        "num_layers": num_layers,
        "n_vocab": n_vocab,
        "lr": lr,
        "seed": seed,
    }

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
        batch_first=batch_first,
        norm_first=norm_first,
        num_layers=num_layers,
        n_vocab=n_vocab,
    )

    log.info("Model: ", model)
    print(model)

    if mode == "train":
        device = accelerator.device

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_dataloader = DataLoader(
            dataset["train"], shuffle=True, batch_size=batch_size
        )
        eval_dataloader = DataLoader(
            dataset["test"], shuffle=False, batch_size=batch_size
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
        ]

        global_step = 0
        for _ in (n_bar := tqdm(range(epochs), desc="Epochs", position=0, leave=False)):
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

                predictions, references = accelerator.gather_for_metrics(
                    (output, target)
                )

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
                predictions, references = accelerator.gather_for_metrics(
                    (output, target)
                )
                for metric in metrics:
                    metric.add_batch(
                        predictions=predictions.argmax(dim=-1), references=references
                    )

            eval_metrics = compute_metrics(metrics, prefix="val")
            accelerator.log(eval_metrics, step=global_step)

            n_bar.set_postfix({"val/acc": eval_metrics["val/accuracy"]})

        accelerator.end_training()

    else:
        raise NotImplementedError


if __name__ == "__main__":
    fire.Fire(main)
