"""Generates data for the group sequence prediction task."""

import os
import random
from functools import reduce
from itertools import product
from pathlib import Path

import fire
import polars as pl
import pyrootutils
from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def group_reduce(lhs: str | int, rhs: int, G) -> int:  # noqa: N803
    """Reduce a sequence of group elements to a single element."""
    if isinstance(lhs, str):
        prod = G.op(lhs, G.elements[rhs])
    else:
        prod = G.op(G.elements[lhs], G.elements[rhs])

    return G.elements.index(prod)


def generate_group(g: (str, int)) -> FiniteAlgebra:
    """Generate an group from a string identifier."""
    if g[0] == "S":
        return generate_symmetric_group(g[1])
    elif g[0] == "Z":
        return generate_cyclic_group(g[1])
    elif g[0] == "A":
        s_n = generate_symmetric_group(g[1])
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{g[1]}"
        return a_n
    else:
        raise ValueError("Group must be one of S, Z, or A")


def main(
    group: str,
    k: int | list[int] = 10,
    samples: int | None = None,
    data_dir: str | Path = PROJECT_ROOT / "data",
    seed: int = random.randint(0, 1_000_000),
    overwrite: bool = False,
):
    """Generate data for the group sequence prediction task."""
    data_path = data_dir / f"{group}={k}.csv"
    if data_path.exists() and not overwrite:
        print(
            f"Data already exists at {data_path}. Use `--overwrite` to regenerate file."
        )
        return

    random.seed(seed)
    print(f"Using seed {seed}")

    group_ids = [(g[0], int(g[1:])) for g in group.split("_x_")]
    for g in group_ids:
        assert g[0] in ["S", "Z", "A"], "Groups must be one of S, Z, or A"
    group_list = [generate_group(g) for g in group_ids]
    group_prod = reduce(lambda x, y: x * y, group_list)

    num_elements = len(group_prod.elements)
    num_unique_sequences = num_elements**k

    if samples is None:
        print(
            f"Generating all {num_elements} ^ {k} = " f"{num_elements ** k} sequences."
        )
        print("Output data will not be shuffled.")

        sequences = product(range(num_elements), repeat=k)

    else:
        if samples > num_unique_sequences:
            print(
                f"Warning: {samples} > {num_unique_sequences}. I will only"
                f" generate {num_unique_sequences} examples."
            )
            samples = num_unique_sequences
        print(f"Randomly sampling {samples} sequences.")
        sequences = set()
        while len(sequences) < samples:
            sequences.add(tuple(random.choices(range(num_elements), k=k)))
        sequences = list(sequences)

    examples = []
    for seq in sequences:
        acc = 0
        scanned = [acc := group_reduce(lhs=acc, rhs=x, G=group_prod) for x in seq]
        examples.append(
            {
                "seed": seed,
                "input": " ".join(map(str, seq)),
                "target": " ".join(map(str, scanned)),
            }
        )
    ex_df = pl.from_dicts(examples)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f"Writing data to `{data_path}`")
    ex_df.write_csv(data_path)


if __name__ == "__main__":
    fire.Fire(main)
