import os
import random
import sys
from functools import partial, reduce
from itertools import product
from pathlib import Path
from typing import List

import fire
import polars as pl
import pyrootutils

# This package is not configured properly, I think.
# Without this line, all methods & submodules will fail to import
sys.path.append("./abstract_algebra/src")

from abstract_algebra.src.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def group_reduce(lhs: str | int, rhs: int, G) -> int:
    if isinstance(lhs, str):
        prod = G.op(lhs, G.elements[rhs])
    else:
        prod = G.op(G.elements[lhs], G.elements[rhs])

    return G.elements.index(prod)


def generate_group(g: (str, int)) -> FiniteAlgebra:
    if g[0] == "S":
        return generate_symmetric_group(g[1])
    elif g[0] == "Z":
        return generate_cyclic_group(g[1])
    elif g[0] == "A":
        s_n = generate_symmetric_group(g[1])
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A_{g[1]}"
        return a_n
    else:
        raise ValueError("Group must be one of S, Z, or A")


def main(
    num_examples: int | None = None,
    seq_length: int | List[int] = 10,
    group: str = "S5",
    data_dir: str | Path = PROJECT_ROOT / "data",
    seed: int = random.randint(0, 1_000_000),
):

    random.seed(seed)

    group_ids = [(g[0], int(g[1:])) for g in group.split("_x_")]
    for g in group_ids:
        assert g[0] in ["S", "Z", "A"], "Groups must be one of S, Z, or A"
    group_list = [generate_group(g) for g in group_ids]
    group_prod = reduce(lambda x, y: x * y, group_list)

    num_elements = len(group_prod.elements)

    if num_examples is None:
        num_possible_sequences = num_elements**seq_length
        print(
            f"Generating {num_elements} ^ {seq_length} = {num_possible_sequences} sequences."
        )
        print("Output data will not be shuffled.")

        sequences = product(range(num_elements), repeat=seq_length)

    else:
        print(
            f"Randomly sampling {num_examples}/{num_elements ** seq_length} sequences."
        )

        sequences = product(range(num_elements), repeat=seq_length)
        sequences = random.choices(list(sequences), k=num_examples)

    examples = []
    for seq in sequences:
        examples.append(
            {
                "length": seq_length,
                "input": " ".join(map(str, seq)),
                "target": str(reduce(partial(group_reduce, G=group_prod), seq)),
            }
        )
    ex_df = pl.from_dicts(examples)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    data_path = data_dir / f"{group_prod.name}={seq_length}.csv"
    print(f"Writing data to `{data_path}`")
    ex_df.write_csv(data_path)


if __name__ == "__main__":
    fire.Fire(main)
