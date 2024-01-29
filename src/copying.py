"""Prefix and suffix copying task."""

import os
import random
from pathlib import Path
from random import randint
from typing import List, Tuple

import fire
import polars as pl
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


class CopyingTask:
    """Generates prefix and suffix copying examples.

    The task is to accept or reject a sequence drawn from the language
    L := a^n{bc}*{bc} (prefix) or L := {bc}*{bc}a^n (suffix), where the final
    character in {bc} is the same as the nth character in the {bc}* group. The
    sequence a^n is known as the index.
    """

    def __init__(self, vocab_size: int, data_length: int, prefix: bool = True):  # noqa D107
        self.vocab = list(range(1, vocab_size + 1))
        self.data_length = data_length
        self.prefix = prefix

    def random_example(self) -> Tuple[List[int], int]:
        """Generate a single random example."""
        idx = random.randint(0, self.data_length - 1)
        tokens = []
        pointer = (0 for _ in range(idx))
        data = [random.choice(self.vocab) for _ in range(self.data_length)]

        if self.prefix:
            tokens.extend(pointer)
            tokens.extend(data)
        else:
            tokens.extend(data)
            tokens.extend(pointer)

        return tokens, data[idx]


def main(
    data_dir: str | Path = PROJECT_ROOT / "data",
    vocab_size: int = 2,
    k: int = 20,
    type: str = "prefix",
    samples: int = 10_000,
    seed: int = randint(0, 2**32 - 1),
    overwrite: bool = False,
):
    """Interface for generating prefix and suffix copying examples."""
    data_path = data_dir / f"copying={type}-{vocab_size}-{k}.csv"
    if data_path.exists() and not overwrite:
        print(
            f"Data already exists at {data_path}. Use `--overwrite` to regenerate file."
        )
        return

    assert type in ["prefix", "suffix"], "type must be one of `prefix` or `suffix`"

    random.seed(seed)
    print(f"Uising seed {seed}")

    task = CopyingTask(vocab_size=vocab_size, data_length=k, prefix=type == "prefix")

    print(f"Generating {samples} samples...")

    examples = []
    for _ in range(samples):
        example = task.random_example()
        examples.append(
            {
                "seed": seed,
                "input": " ".join(map(str, example[0])),
                "target": f"{example[1]}",
            }
        )

    ex_df = pl.from_dicts(examples)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print(f"Writing data to `{data_path}`")
    ex_df.write_csv(data_path)


if __name__ == "__main__":
    fire.Fire(main)
