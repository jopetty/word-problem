"""Main entry point for training models."""

import logging
from collections.abc import Callable
from enum import StrEnum
from functools import partial
from pathlib import Path
from pprint import pformat
from random import randint

import fire
import polars as pl
import pyrootutils
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from ordered_set import OrderedSet
from sfirah.metrics import ce_loss, sequence_accuracy, token_accuracy
from sfirah.mlp import MLPSequenceClassifier
from sfirah.transformers import EncoderSequenceClassifier, EncoderTokenClassifier
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

    PAD = "[PAD]"
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


def pad_collate(
    samples: list[dict[str, Tensor]], pad_token_id: int
) -> dict[str, Tensor]:
    """Collate function for DataLoader.

    Performs channel-wise padding of the inputs and targets.
    """
    # Only pad `labels` if len(labels) > 1,
    channels_to_pad = ["input_ids"]
    if samples[0]["labels"].dim() > 0:
        channels_to_pad.append("labels")

    max_lens = {}
    for c in channels_to_pad:
        max_lens[c] = max([s[c].shape[0] for s in samples])

    for s in samples:
        for c in channels_to_pad:
            if max_lens[c] > s[c].shape[0]:
                s[c] = F.pad(s[c], (0, max_lens[c] - s[c].shape[0]), value=pad_token_id)

    collated = {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
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
    tokenized.pop("attention_mask", None)

    # If output is not supervised (e.g., for MLPs) then we only keep the final target
    # value since its sequence classification, not token classification.
    tokenized["labels"] = tokenizer(example["target"], return_tensors="pt")["input_ids"]
    if not supervised:
        tokenized["labels"] = tokenized["labels"][:, -1]

    return tokenized


def get_dataset(
    group: str,
    k: int,
    strict_len: bool,
    train_size: float,
    data_dir: str | Path,
    supervised: bool = True,
) -> dict:
    """Construct dataset."""
    assert train_size > 0 and train_size <= 1, "`train_size` must be in (0,1]"

    if strict_len:
        assert k > 1, "`k` must be at least 2"
        data_paths = [data_dir / f"{group}={i}.csv" for i in [2, k]]
        data_paths = list(OrderedSet(data_paths))
        if not data_paths[0].exists():
            raise FileNotFoundError(f"You must have data for {group}={2}.")
        log.info("Constructing dataset from:")
        log.info("  " + "\n  ".join(map(str, data_paths)))
    else:
        data_paths = [data_dir / f"{group}={i}.csv" for i in range(2, k + 1)]
        if not data_paths[0].exists():
            raise FileNotFoundError(f"You must have data for {group}=2.")
        data_paths = [p for p in data_paths if p.exists()]
        data_paths = list(OrderedSet(data_paths))
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
        long_data = (
            load_dataset("csv", data_files=map(str, data_paths[1:]), split="all")
            .remove_columns(["seed"])
            .map(tokenize_map, batched=True)
            .remove_columns(["input", "target", "token_type_ids"])
        )

        if train_size < 1:
            dataset = long_data.train_test_split(train_size=train_size)
            dataset["train"] = concatenate_datasets([dataset["train"], pair_data])
        else:
            dataset = concatenate_datasets([long_data, pair_data])

    return {
        "dataset": dataset.with_format("torch"),
        "tokenizer": tokenizer,
        "n_vocab": tokenizer_base.get_vocab_size(with_added_tokens=True),
    }


def detach_and_pad(
    data: list[(Tensor, Tensor)], pad_token_id: int
) -> dict[str, Tensor]:
    """Detach tensors from accelerator and pad them to be the same length."""
    preds = [d[0].cpu().detach() for d in data]
    tgts = [d[1].cpu().detach() for d in data]

    max_pred_len = max([p.shape[1] for p in preds])
    min_pred_len = min([p.shape[1] for p in preds])
    max_tgt_len = max([t.shape[-1] for t in tgts])
    min_tgt_len = min([t.shape[-1] for t in tgts])

    if max_pred_len != min_pred_len:
        for idx, p in enumerate(preds):
            pred_pad_size = max_pred_len - p.shape[1]
            if pred_pad_size > 0:
                padding_logits = torch.ones_like(p[:, [0], :]) * float("-inf")
                padding_logits[:, :, pad_token_id] = 1.0

                padding_logits = torch.cat([padding_logits] * pred_pad_size, dim=1)

                for _ in range(pred_pad_size):
                    p = torch.cat((padding_logits, p), dim=1)

                preds[idx] = p

    # if targets are single values per sequence (i.e., non-supervised) then we
    # don't need to worry about padding anything
    if (max_tgt_len != min_tgt_len) and (tgts[0].dim() > 1):
        for idx, t in enumerate(tgts):
            tgt_pad_size = max_tgt_len - t.shape[-1]
            if tgt_pad_size > 0:
                t = F.pad(t, (tgt_pad_size, 0), mode="constant", value=pad_token_id)
                tgts[idx] = t

    preds = torch.cat(preds, dim=0)
    tgts = torch.cat(tgts, dim=0)

    return {"predictions": preds, "targets": tgts}


def compute_metrics(
    data: list[(Tensor, Tensor)],
    tokenizer: PreTrainedTokenizerFast,
    metric_fns: dict[str, Callable] = {
        "loss": ce_loss,
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
    },
    prefix: str | None = None,
) -> dict:
    """Compute metrics."""
    values_dict = {}

    data = detach_and_pad(data, pad_token_id=tokenizer.pad_token_id)
    predicted_logits = data["predictions"]
    target_tokens = data["targets"]

    prefix_str = "" if prefix is None else f"{prefix}/"
    for metric_name, metric_fn in metric_fns.items():
        values_dict[prefix_str + metric_name] = metric_fn(
            predicted_logits, target_tokens, tokenizer.pad_token_id
        )

    return values_dict


def train_mlp(
    # Data parameters
    group: str,
    data_dir: Path = PROJECT_ROOT / "data",
    # Model parameters
    d_model: int = 512,
    d_ff: int = 2048,
    activation: str = "relu",
    dropout: float = 0.1,
    n_layers: int = 1,
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
        k=2,
        strict_len=True,
        train_size=1.0,
        data_dir=data_dir,
        supervised=False,
    )
    dataset = datadict["dataset"]
    n_vocab = datadict["n_vocab"]
    tokenizer = datadict["tokenizer"]
    pad_token_id = datadict["tokenizer"].pad_token_id
    collate_fn = partial(pad_collate, pad_token_id=pad_token_id)

    # Set up logger
    project_hps = {
        "group": group,
        "d_model": d_model,
        "d_ff": d_ff,
        "activation": activation,
        "dropout": dropout,
        "n_layers": n_layers,
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

    accelerator.init_trackers(
        project_name,
        config=project_hps,
    )

    log.info(f"Config: {pformat(project_hps)}")
    log.info(f"Dataset: {dataset}")

    model = MLPSequenceClassifier(
        d_model=d_model,
        d_ff=d_ff,
        activation=activation,
        dropout=dropout,
        n_layers=n_layers,
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
        collate_fn=collate_fn,
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

            loss = F.cross_entropy(output, target, ignore_index=pad_token_id)
            accelerator.backward(loss)
            optimizer.step()

            predictions, references = accelerator.gather_for_metrics((output, target))
            train_metric_data.append((predictions, references))

            t_bar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_metrics = compute_metrics(
            train_metric_data, metric_fns=metrics, prefix="train", tokenizer=tokenizer
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
    k: int = 3,
    strict_len: bool = True,
    train_size: float = 0.8,
    tagging: bool = True,
    # Model parameters
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
    activation: str = "gelu",
    layer_norm_eps: float = 1e-5,
    norm_first: bool = False,
    n_layers: int = 1,
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
    causal: bool = True,
    gradient_clip: float | None = None,
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
        group=group,
        k=k,
        strict_len=strict_len,
        train_size=train_size,
        data_dir=data_dir,
        supervised=tagging,
    )
    dataset = datadict["dataset"]
    n_vocab = datadict["n_vocab"]
    tokenizer = datadict["tokenizer"]
    collate_fn = partial(pad_collate, pad_token_id=tokenizer.pad_token_id)

    # Set up logger
    project_hps = {
        "activation": activation,
        "batch_size": batch_size,
        "betas": (beta1, beta2),
        "bias": bias,
        "causal": causal,
        "compile": compile,
        "d_model": d_model,
        "d_ff": d_ff,
        "dropout": dropout,
        "epochs": epochs,
        "eps": op_eps,
        "group": group,
        "gradient_clip": gradient_clip,
        "k": k,
        "layer_norm_eps": layer_norm_eps,
        "lr": lr,
        "n_heads": n_heads,
        "norm_first": norm_first,
        "n_layers": n_layers,
        "n_vocab": n_vocab,
        "seed": seed,
        "strict_len": strict_len,
        "tagging": tagging,
        "train_size": train_size,
        "weight_decay": weight_decay,
        "weight_scale": weight_scale,
        "weight_sharing": weight_sharing,
    }

    accelerator.init_trackers(
        project_name,
        config=project_hps,
    )

    log.info(f"Config: {pformat(project_hps)}")
    log.info(f"Dataset: {dataset}")

    # Construct model
    if tagging:
        model = EncoderTokenClassifier(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            n_layers=n_layers,
            weight_sharing=weight_sharing,
            weight_scale=weight_scale,
            n_vocab=n_vocab,
            batch_first=True,
            bias=bias,
        )
    else:
        model = EncoderSequenceClassifier(
            cl_dim=1,
            cl_index=0,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            n_layers=n_layers,
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
    if train_size < 1:
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        eval_dataloader = DataLoader(
            dataset["test"],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        eval_dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    metric_fns = {
        "loss": ce_loss,
        "sequence_accuracy": token_accuracy,
    }

    if tagging:
        metric_fns["sequence_accuracy"] = sequence_accuracy
        metric_fns["token_accuracy"] = token_accuracy

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
            target = batch["labels"]

            if causal:
                mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    source.shape[1], device=device
                )
                output = model(source, mask=mask, is_causal=True)
            else:
                output = model(source)

            predictions, references = accelerator.gather_for_metrics((output, target))
            train_data.append((predictions, references))

            target = target.flatten()
            output = output.flatten(end_dim=-2)
            loss = F.cross_entropy(output, target)

            if global_step % 100 == 0:
                log.debug(f"preds: {predictions.argmax(dim=-1)}")
                log.debug(f"trgts: {references}")

            accelerator.backward(loss)

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip, norm_type=2.0
                )

            optimizer.step()

            t_bar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_metrics = compute_metrics(
            train_data, prefix="train", tokenizer=tokenizer, metric_fns=metric_fns
        )
        accelerator.log(train_metrics, step=global_step)

        model.eval()
        eval_data = []
        for batch in tqdm(eval_dataloader, desc="Eval", position=1, leave=False):
            source = batch["input_ids"]
            target = batch["labels"]
            with torch.no_grad():
                if causal:
                    mask = torch.nn.Transformer.generate_square_subsequent_mask(
                        source.shape[1], device=device
                    )
                    output = model(source, mask=mask, is_causal=True)
                else:
                    output = model(source)

            predictions, references = accelerator.gather_for_metrics((output, target))

            eval_data.append((predictions, references))

        eval_metrics = compute_metrics(
            eval_data, prefix="val", tokenizer=tokenizer, metric_fns=metric_fns
        )
        accelerator.log(eval_metrics, step=global_step)
        accelerator.log({"epoch": epoch}, step=global_step)

        n_bar.set_postfix({"val/acc": f"{eval_metrics['val/sequence_accuracy']:.3f}"})

    log.info(eval_metrics)
    accelerator.end_training()


if __name__ == "__main__":
    fire.Fire()
