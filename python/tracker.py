class PythonTracker:

    def __init__(self, eval_context, history, mode):
        self.eval_context = eval_context
        self.history = history
        self.mode = mode

    @classmethod
    def initialize(cls, *values, mode="tuple"):
        eval_context = {f"x{i}": v for i, v in enumerate(values)}
        history = [f"x{i} = {v}" for i, v in enumerate(values)]
        return cls(eval_context, history, mode)

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