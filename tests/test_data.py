"""Test dataset loading and preprocessing."""


import logging
import unittest

import pyrootutils
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger

from main import get_dataset, pad_collate

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
log = get_logger(__name__)


PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


class TestData(unittest.TestCase):  # noqa: D101
    def test_pad_collate_supervised(self):  # noqa: D102
        _ = Accelerator()

        batch = [
            {
                "input_ids": [7, 1, 2, 3],
                "labels": [7, 1, 2, 3],
                "attention_mask": [False, False, False, False],
            },
            {
                "input_ids": [7, 1, 2],
                "labels": [7, 1, 2],
                "attention_mask": [False, False, False],
            },
            {
                "input_ids": [7, 1, 2, 3, 4, 5],
                "labels": [7, 1, 2, 3, 4, 5],
                "attention_mask": [False, False, False, False, False, False],
            },
        ]

        batch = [{k: torch.tensor(v) for k, v in s.items()} for s in batch]
        collated_batch = pad_collate(batch)

        self.assertEqual(collated_batch["input_ids"].shape, (3, 6))
        self.assertEqual(collated_batch["labels"].shape, (3, 6))

        # check that the first tokens have been preserved
        for seq in range(collated_batch["input_ids"].shape[0]):
            self.assertEqual(
                collated_batch["input_ids"][seq, 0], batch[seq]["input_ids"][0]
            )

        # check that the last tokens have been preserved
        for seq in range(collated_batch["input_ids"].shape[0]):
            self.assertEqual(
                collated_batch["input_ids"][seq, -1], batch[seq]["input_ids"][-1]
            )

        # check that padding has been applied
        self.assertEqual(collated_batch["input_ids"][0, 1], 0)

    def test_load_supervised_data(self):  # noqa: D102
        _ = Accelerator()

        k = 3
        group = "Z4"
        data_dir = PROJECT_ROOT / "data"
        max_len = None
        train_size = 0.8
        supervised = True

        get_dataset(
            data_dir=data_dir,
            group=group,
            k=k,
            max_len=max_len,
            train_size=train_size,
            supervised=supervised,
        )


if __name__ == "__main__":
    unittest.main()
