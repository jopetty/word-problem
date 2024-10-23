import argparse
import logging
import transformers
import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

os.environ["WANDB_PROJECT"] = "log-depth"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--phase-name", type=str, nargs="+", required=True)
    parser.add_argument("--train-path", type=str, nargs="+", required=True)
    parser.add_argument("--val-path", type=str, nargs="+", required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--logs-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=0)
    # These parameters are pretty stable/not worth changing.
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=60)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 5, 10, 100, 1000])
    parser.add_argument("--eps", type=float, nargs="+", default=[0.05])
    return parser.parse_args()

class GroupDataset(Dataset):
    """Dataset to load group data saved as a CSV."""

    def __init__(self, csv, tokenizer):
        self.csv = csv
        self.tokenizer = tokenizer
    
    @classmethod
    def from_csv(cls, path, tokenizer):
        return cls(pd.read_csv(path), tokenizer)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        input_text = self.csv.input[idx]
        target_text = self.csv.target[idx]

        input_ids = self.tokenizer(input_text).input_ids
        labels = [int(x) for x in target_text.split()]
        if len(input_ids) != len(labels):
            breakpoint()

        assert len(input_ids) == len(labels), f"Input and target lengths do not match: {len(input_ids)} != {len(labels)}"

        return {
            # Pythia tokenizer seems to correctly map each integer to its own token.
            "input_ids": self.tokenizer(input_text).input_ids,
            # Convert to list of integers in range [0, group_size).
            "labels": [int(x) for x in target_text.split()],
        }

class Evaluator:
    def __init__(self, indices: list[int], eps: list[float]):
        self.indices = indices
        self.eps = eps

    def compute_metrics(self, eval_preds) -> dict:
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids
        top_preds = predictions.argmax(axis=-1)
        accs_by_idx = np.mean(top_preds == labels, axis=0)

        results = {}
        for idx in self.indices:
            if idx >= len(accs_by_idx):
                continue
            results[f"acc[{idx}]"] = accs_by_idx[idx].item()
        for eps in self.eps:
            failures, = np.where(accs_by_idx < 1 - eps)
            results[f"n@{eps}"] = failures.min().item() if len(failures) > 0 else len(accs_by_idx)
        return results

class WandbStepCallback(TrainerCallback):
    def __init__(self, global_step: int):
        self.global_step = global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Adjust the step to continue from phase 1
            adjusted_step = self.global_step + state.global_step
            logs['step'] = adjusted_step
            wandb.log(logs)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    evaluator = Evaluator(args.indices, args.eps)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=args.group_size)
    if torch.cuda.is_available():
        model.to("cuda")

    global_step = 0
    for phase_name, train_path, val_path in zip(args.phase_name, args.train_path, args.val_path):
        run_name = args.model.split("/")[-1]
        log.info(f"Training {run_name} on {train_path}...")
        training_args = TrainingArguments(
            output_dir=args.results_dir,
            logging_dir=args.logs_dir,
            num_train_epochs=args.n_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            weight_decay=0.01,
            report_to="wandb",
            run_name=run_name,
            logging_steps=args.log_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            warmup_steps=args.warmup_steps,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=GroupDataset.from_csv(train_path, tokenizer),
            eval_dataset=GroupDataset.from_csv(val_path, tokenizer),
            compute_metrics=evaluator.compute_metrics,
        )
        trainer.add_callback(WandbStepCallback(global_step))
        trainer.train()
        global_step = trainer.state.global_step
        wandb.log({"phase": phase_name})

if __name__ == "__main__":
    main(parse_args())