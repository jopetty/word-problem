from typing import Union
from accelerate.logging import get_logger
import fire
import pyrootutils
from tqdm.auto import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
import polars as pl
import torch
import numpy as np
from torch import nn, optim, Tensor
import torch.nn.functional as F
from datasets import load_dataset
from pathlib import Path
from accelerate import Accelerator
from evaluate import load as load_metric
from random import randint
from accelerate.utils import set_seed

log = get_logger(__name__, log_level="INFO")

PROJECT_ROOT = path = pyrootutils.find_root(
  search_from=__file__, 
  indicator=".project-root")

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
    self.register_buffer('pe', pe)

  def forward(self, x: Tensor) -> Tensor:
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    x = x + self.pe[:x.size(0)]
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
      norm_first=norm_first)
    self.embedding = nn.Embedding(n_vocab, d_model)
    self.pos_enc = PositionalEncoding(d_model, dropout)
    self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    self.classifier = nn.Linear(d_model, n_vocab)
  
  def forward(self, x):
    x = self.pos_enc(self.embedding(x))
    x = self.encoder(x)
    # print(x.shape)
    x = x.mean(dim=1)
    # print(x.shape)
    logits = self.classifier(x)
    # print(logits.shape)
    # raise SystemExit
    return logits

def compute_metrics(targets, pred_logits):

  acc_metric = load_metric("accuracy")
  predictions = pred_logits.argmax(dim=-1)
  return acc_metric.compute(predictions=predictions, references=targets)

def main(
  data: str = "s5_1",
  data_dir: Union[str, Path] = PROJECT_ROOT / "data",
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
):
  
  set_seed(seed)

  seq_len = data.split("_")[1]

  accelerator = Accelerator(log_with="wandb")
  # accelerator = Accelerator()
  
  log.setLevel(log_level)
  
  assert mode in ["train", "test"], "mode must be either 'train' or 'test'"

  # Load dataset
  data_path = str(data_dir / f"{data}.csv")
  dataset = load_dataset("csv", data_files=data_path, split="all")
  dataset = dataset.remove_columns(["length"])

  # "Tokenize" data by converting inputs back into list of integers
  #   instead of strings to make parsing the CSV easier
  dataset = dataset.map(lambda ex: {
    "input": list(map(int, str(ex["input"]).split())), 
    "target": int(ex["target"])
  })
  dataset = dataset.with_format(type="torch")

  assert train_split > 0 and train_split < 1, "train_split must be between 0 and 1"
  dataset = dataset.train_test_split(train_size=train_split)

  log.info("Dataset: ", dataset)
  print(dataset)

  # Figure out the number of unique tokens in the dataset
  n_vocab = pl.read_csv(data_path).with_columns(
    pl.concat_str([pl.col("input"), pl.col("target")], separator=" ").alias("merged")
  ).select(pl.col("merged").map_batches(
    lambda x: x.str.split(" ")
  )).explode("merged").unique().shape[0]

  # Set up logger
  project_name = "depth-test"
  project_hps = {
    "sequence_length": seq_len,
    "dataset": data,
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
    n_vocab=n_vocab)
  
  log.info("Model: ", model)
  print(model)

  if mode == "train":
    device = accelerator.device

    model = model.to(device)
    optimizer = optim.Adam(
      model.parameters(),
      lr=lr)
    train_dataloader = DataLoader(
      dataset["train"], 
      shuffle=True, 
      batch_size=batch_size)

    model, optimizer, train_dataloader = accelerator.prepare(
      model, 
      optimizer, 
      train_dataloader)
    
    # progress_bar = tqdm(
    #   train_dataloader, 
    #   disable=not accelerator.is_local_main_process,
    #   desc="Batch")
    
    
    model.train()
    for _ in range(epochs):
      for batch in tqdm(train_dataloader, desc="Batch"):

        optimizer.zero_grad()

        source = batch["input"]
        target = batch["target"]
        output = model(source)

        loss = F.cross_entropy(output, target)
        metrics = compute_metrics(target, output)
        metrics["loss"] = loss.item()
        accelerator.log(metrics)

        accelerator.backward(loss)
        optimizer.step()
        # progress_bar.set_postfix({
        #   "loss: ": f"{metrics['loss']:.2f}",
        #   "acc": f"{metrics['accuracy']:.2f}"
        # })
      print({"loss": loss})
    accelerator.end_training()

  else:
    raise NotImplementedError

if __name__ == "__main__":
  fire.Fire(main)