"""Generate data for chess/Python state tracking experiments."""

import json
import random
from typing import Any, Tuple, List
import argparse
import random

from state_tracking.python import PythonTracker
from state_tracking.chess import BoardTracker

class DataGenerator:

    def __init__(self, new_tracker, n_items):
        self.new_tracker = new_tracker
        self.n_items = n_items
    
    def sample_transposition_instance(self, length: int) -> Tuple[List[str], Any]:
        """Function to sample a sequence of transpositions and final state.

        Returns the list of actions and the final state. The type of the state is:
          * Chess: A numpy array representing the board
          * Python: A dict[str, bool] of variable assignments
        """
        tracker = self.new_tracker(self.n_items)

        for _ in range(length):
            i = random.randint(0, self.n_items - 1)
            j = random.randint(0, self.n_items - 1)
            tracker.transpose(i, j)

        return {
            "history": tracker.get_history(),
            "state": tracker.get_state(),
        }


def main(args):
    if args.problem == "chess":
        new_tracker = lambda n_items: BoardTracker.queen_rook_permutations(n_items)
    elif args.problem == "python":
        new_tracker = lambda n_items: PythonTracker.random_init(n_items)
    else:
        raise ValueError("unknown problem:", args.problem)

    generator = DataGenerator(new_tracker, args.n_items)
    for _ in range(args.n_samples):
        blob = generator.sample_transposition_instance(args.length)
        print(json.dumps(blob))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", choices=["python", "chess"])
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--n_items", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--no_piece_type", action="store_true")
    parser.add_argument("--no_source", action="store_true")
    parser.add_argument("--no_target", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())