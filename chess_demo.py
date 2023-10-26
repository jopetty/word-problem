import random

from chessboard import BoardTracker

board = BoardTracker.queen_rook_permutations()
for _ in range(10):
    source = random.randint(0, 4)
    target = source
    while target == source:
        target = random.randint(0, 4)
    board.transpose(source, target)
print([move.format(piece_type=False) for move in board.history])
