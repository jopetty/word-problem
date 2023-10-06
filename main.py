"""Main entry point for training models."""

import logging
from collections.abc import Callable
from enum import StrEnum
from functools import partial
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
from sfirah.metrics import ce_loss, sequence_accuracy, token_accuracy
from sfirah.mlp import MLPSequenceClassifier
from sfirah.transformers import EncoderTokenClassifier
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

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


class SpecialTokens(StrEnum):
    """Special tokens for tokenizer.

    Uses the group identity element '0' as the padding token, allowing us to pad
    supervised tagging tasks easily. Note that for this to make semantic sense,
    we must use left-padding. Aside from PAD and BOS, the other tokens are not used,
    but it's useful to have them to prevent PreTrainedFastTokenizer from complaining
    that they are missing.
    """

    PAD = "0"  # Use the group identity token as [PAD]
    BOS = "[BOS]"
    UNK = "[UNK]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    CLS = "[CLS]"
    MASK = "[MASK]"

    @classmethod
    def values(cls):
        """Return a list of the string values of each special token."""
        return list(map(lambda c: c.value, cls))

    @property
    def index(self):
        """Return the index of the token in the vocabulary.

        Used to get the index of the PAD token when directly modifying tensors.
        """
        return SpecialTokens.values().index(self.value)


def pad_collate(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate function for DataLoader.

    Performs channel-wise padding of the inputs and targets.
    """
    # Only pad `labels` if len(labels) > 1,
    channels_to_pad = ["input_ids", "attention_mask"]
    if samples[0]["labels"].dim() > 0:
        channels_to_pad.append("labels")

    max_lens = {}
    for c in channels_to_pad:
        max_lens[c] = max([s[c].shape[0] for s in samples])

    for s in samples:
        for c in channels_to_pad:
            if max_lens[c] > s[c].shape[0]:
                if s[c].dtype == torch.bool:
                    s[c] = F.pad(
                        s[c],
                        (max_lens[c] - s[c].shape[0], 0),
                        value=False,
                    )
                else:
                    bos = s[c][[0]]
                    rest = s[c][1:]

                    padded_rest = F.pad(
                        rest,
                        (max_lens[c] - s[c].shape[0], 0),
                        value=SpecialTokens.PAD.index,
                    )
                    # padded = torch.cat((bos, padded_rest), dim=0)
                    # print(s[c])
                    # print(padded)

                    s[c] = torch.cat((bos, padded_rest), dim=0)
                    # raise RuntimeError("You need to check this works!!")

    collated = {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
        "attention_mask": torch.zeros(
            (samples[0]["input_ids"].shape[0], samples[0]["input_ids"].shape[0]),
            dtype=torch.bool,
        ),
    }

    return collated


def tokenize(
    example: dict[str, Tensor],
    tokenizer: PreTrainedTokenizerFast,
    supervised: bool,
) -> dict[str, Tensor]:
    """Tokenize inputs."""
    tokenized = tokenizer(
        example["input"],
        return_tensors="pt",
    )

    # If output is not supervised (e.g., for MLPs) then we only keep the final target
    # value since its sequence classification, not token classification.
    tokenized["labels"] = tokenizer(example["target"], return_tensors="pt")["input_ids"]
    if not supervised:
        tokenized["labels"] = tokenized["labels"][:, -1]

    # We need to overwrite the attention mask to allow attending to the padding
    # tokens, since we use the group identity element as padding. We also need
    # to convert it to a boolean tensor.
    tokenized["attention_mask"] = torch.zeros_like(
        tokenized["input_ids"], dtype=torch.bool
    )

    return tokenized


def get_dataset(
    group: str,
    max_len: int | None,
    k: int | None,
    train_size: float,
    data_dir: str | Path,
    supervised: bool = True,
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

    # All unique tokens can be found by looking at the k=2 inputs. We create a
    # a dictionary mapping each token to its index in the vocabulary and use this
    # to construct the tokenizer.
    unique_tokens = (
        pl.read_csv(data_paths[0])
        .select(pl.col("input").map_batches(lambda x: x.str.split(" ")))
        .explode("input")
        .unique()["input"]
        .to_list()
    )
    unique_tokens = {t: int(t) for t in unique_tokens}

    tokenizer_base = Tokenizer(WordLevel())
    tokenizer_base.pre_tokenizer = WhitespaceSplit()
    tokenizer_base.add_tokens(sorted(list(unique_tokens.keys()), key=lambda x: int(x)))
    tokenizer_base.add_special_tokens(SpecialTokens.values())
    tokenizer_base.post_processor = TemplateProcessing(
        single=f"{SpecialTokens.BOS} $A",
        special_tokens=[
            (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS))
        ],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_base,
        bos_token=SpecialTokens.BOS.value,
        unk_token=SpecialTokens.UNK.value,
        eos_token=SpecialTokens.EOS.value,
        sep_token=SpecialTokens.SEP.value,
        cls_token=SpecialTokens.CLS.value,
        mask_token=SpecialTokens.MASK.value,
        pad_token=SpecialTokens.PAD.value,
    )
    tokenizer.padding_side = "left"
    tokenize_map = partial(tokenize, tokenizer=tokenizer, supervised=supervised)

    # Construct dataset
    if len(data_paths) == 1:
        dataset = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )
        if train_size < 1:
            dataset = dataset.train_test_split(train_size=train_size)
    else:
        pair_data = (
            load_dataset("csv", data_files=str(data_paths[0]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )
        long_data = [
            (
                load_dataset("csv", data_files=str(p), split="all")
                .remove_columns(["seed"])
                .map(tokenize_map, batched=True)
                .remove_columns(["input", "target", "token_type_ids"])
            )
            for p in data_paths[1:]
        ]

        dataset = concatenate_datasets(long_data).train_test_split(
            train_size=train_size
        )
        dataset["train"] = concatenate_datasets([dataset["train"], pair_data])

    return {
        "dataset": dataset.with_format("torch"),
        "tokenizer": tokenizer,
        "n_vocab": tokenizer_base.get_vocab_size(with_added_tokens=True),
    }


def compute_metrics(
    data: list[(Tensor, Tensor)],
    metric_fns: dict[str, Callable] = {
        "loss": ce_loss,
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
    },
    prefix: str | None = None,
) -> dict:
    """Compute metrics."""
    values_dict = {}

    # Detach tensors from accelerator to get rid of autograd information
    detached_data = [(d[0].cpu().detach(), d[1].cpu().detach()) for d in data]

    if detached_data[0][0].dim() > 2:
        # Even though each batch has sequences with the same number of tokens,
        # the seq_len of each batch may be different. We need to pad the predictions
        # and targets to be the same length before we concatenate them together.
        seq_len_max = max([d[0].shape[2] for d in detached_data])
        padded_preds = []
        padded_tgts = []
        for d in detached_data:
            pred, tgt = d
            pred_pad_size = seq_len_max - pred.shape[2]
            tgt_pad_size = seq_len_max - tgt.shape[1]

            if pred_pad_size > 0:
                padding = (pred_pad_size, 0)
                pad_logits = torch.ones_like(pred[:, :, [0]]) * float("-inf")
                pad_logits[:, SpecialTokens.PAD.index, :] = 1.0

                bos = pred[:, :, [0]]
                rest = pred[:, :, 1:]

                for _ in range(pred_pad_size):
                    rest = torch.cat((pad_logits, rest), dim=-1)

                padded_pred = torch.cat((bos, rest), dim=-1)
                print(padded_pred)
                print(padded_pred.argmax(dim=1))
                padded_preds.append(padded_pred)
            else:
                padded_preds.append(pred)

            if tgt_pad_size > 0:
                padding = (tgt_pad_size, 0)
                padded_tgt = F.pad(tgt[:, 1:], padding, value=SpecialTokens.PAD.index)
                padded_tgt = torch.cat((tgt[:, [0]], padded_tgt), dim=-1)
                print(padded_tgt)
                padded_tgts.append(padded_tgt)
            else:
                padded_tgts.append(tgt)

        predicted_logits = torch.cat(padded_preds, dim=0)
        target_tokens = torch.cat(padded_tgts, dim=0)
    else:
        predicted_logits = torch.cat([d[0] for d in detached_data], dim=0)
        target_tokens = torch.cat([d[1] for d in detached_data], dim=0)

    prefix_str = "" if prefix is None else f"{prefix}/"
    for metric_name, metric_fn in metric_fns.items():
        values_dict[prefix_str + metric_name] = metric_fn(
            predicted_logits, target_tokens
        )

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
    datadict = get_dataset(
        group=group,
        max_len=None,
        k=2,
        train_size=1.0,
        data_dir=data_dir,
        supervised=False,
    )
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
        d_ff=dim_feedforward,
        activation=activation,
        dropout=dropout,
        n_layers=num_layers,
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

    metrics = {
        "loss": ce_loss,
        "accuracy": token_accuracy,
    }

    global_step = 0
    for epoch in (n_bar := tqdm(range(epochs), desc="Epochs", position=0, leave=False)):
        model.train()
        train_metric_data = []
        for batch in (
            t_bar := tqdm(train_dataloader, desc="Train", position=1, leave=False)
        ):
            global_step += 1
            optimizer.zero_grad()

            source = batch["input_ids"]
            target = batch["labels"]

            output = model(source)

            loss = F.cross_entropy(output, target)
            accelerator.backward(loss)
            optimizer.step()

            predictions, references = accelerator.gather_for_metrics((output, target))
            train_metric_data.append((predictions, references))

            t_bar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_metrics = compute_metrics(
            train_metric_data, metric_fns=metrics, prefix="train"
        )

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
    supervised: bool = True,
    # Model parameters
    d_model: int = 512,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str = "relu",  # TODO: Make "gelu" default
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
    datadict = get_dataset(
        group, max_len, k, train_split, data_dir, supervised=supervised
    )
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

            source = batch["input_ids"]
            mask = batch["attention_mask"]
            target = batch["labels"]

            # print(source.shape, mask.shape, target.shape)
            # print(f"Source shape: {source.shape}")
            # print(f"Source: {source}")

            # print(f"Mask shape: {mask.shape}")
            # print(f"Mask: {mask}")

            # print(f"Target shape: {target.shape}")
            # print(f"Target: {target}")

            output = model(source, mask=mask)

            # print(f"Output shape: {output.shape}")
            # print(f"Output: {output}")

            # raise SystemExit

            loss = F.cross_entropy(output, target)

            predictions, references = accelerator.gather_for_metrics((output, target))

            train_data.append((predictions, references))

            if np.random.random_sample() < 0.01:
                log.debug(f"preds: {predictions.argmax(dim=1)}")
                log.debug(f"trgts: {references}")

            accelerator.backward(loss)
            optimizer.step()

            t_bar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_metrics = compute_metrics(train_data, prefix="train")
        accelerator.log(train_metrics, step=global_step)

        model.eval()
        eval_data = []
        for batch in tqdm(eval_dataloader, desc="Eval", position=1, leave=False):
            source = batch["input_ids"]
            mask = batch["attention_mask"]
            target = batch["labels"]
            with torch.no_grad():
                output = model(source, mask=mask)

            predictions, references = accelerator.gather_for_metrics((output, target))

            eval_data.append((predictions, references))

        eval_metrics = compute_metrics(eval_data, prefix="val")
        accelerator.log(eval_metrics, step=global_step)
        accelerator.log({"epoch": epoch}, step=global_step)

        n_bar.set_postfix({"val/acc": f"{eval_metrics['val/sequence_accuracy']:.3f}"})

    log.info(eval_metrics)
    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire()
