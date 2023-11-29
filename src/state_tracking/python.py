import random

from .tracker import Tracker

BITS = [False, True]


class PythonTracker(Tracker):
    def __init__(self, eval_context, history, mode):
        self.eval_context = eval_context
        self.history = history
        self.mode = mode

    @classmethod
    def initialize(cls, *values, mode="tuple"):
        eval_context = {f"x{i}": v for i, v in enumerate(values)}
        history = [f"x{i} = {v}" for i, v in enumerate(values)]
        return cls(eval_context, history, mode)

    @classmethod
    def random_init(cls, n_items: int = 5, mode="tuple"):
        values = [random.choice(BITS) for _ in range(n_items)]
        return cls.initialize(*values, mode=mode)

    def transpose(self, i, j):
        z = self.eval_context[f"x{i}"]
        self.eval_context[f"x{i}"] = self.eval_context[f"x{j}"]
        self.eval_context[f"x{j}"] = z

        match self.mode:
            case "z":
                self.history.append(f"z = x{i}")
                self.history.append(f"x{i} = x{j}")
                self.history.append(f"x{j} = z")
            case "if":
                self.history.append(f"if x{i} and not x{j}:, x{i}, x{j} = False, True")
                self.history.append(f"if not x{i} and x{j}: x{i}, x{j} = True, False")
            case "tuple":
                self.history.append(f"x{i}, x{j} = x{j}, x{i}")

    def get_history(self):
        return self.history

    def get_state(self):
        return self.eval_context
