from typing import List, Tuple
import random

class CopyingTask:

    def __init__(self, vocab_size: int, data_length: int, prefix: bool = True):
        self.vocab = list(range(1, vocab_size + 1))
        self.data_length = data_length
        self.prefix = prefix
    
    def random_example(self) -> Tuple[List[int], int]:
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


if __name__ == "__main__":
    task = CopyingTask(5, 20, prefix=False)
    for _ in range(5):
        print(task.random_example())