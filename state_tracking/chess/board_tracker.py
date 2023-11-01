import numpy as np

from ..tracker import Tracker
from .move import Move
from .piece_type import PieceType

class BoardTracker(Tracker):

    def __init__(self, size=(8, 8), **fmt_options):
        self.white = True
        self.board = np.ones(size, dtype=np.int32) * PieceType.EMPTY.value
        self.history = []
        self.fmt_options = fmt_options
    
    @classmethod
    def queen_rook_permutations(cls, n_items: int = 5):
        board = cls()
        for j in range(n_items):
            board[0, j] = PieceType.WHITE_QUEEN
        board[0, 0] = PieceType.WHITE_ROOK
        board[0, 7] = PieceType.WHITE_KING

        board[7, 0] = PieceType.BLACK_ROOK
        board[6, 6] = PieceType.WHITE_PAWN
        board[6, 7] = PieceType.WHITE_PAWN
        board[7, 7] = PieceType.BLACK_KING

        return board
    
    def __setitem__(self, index, piece_type):
        self.board[index] = piece_type.value

    def __getitem__(self, index):
        return PieceType(self.board[index])

    def move(self, source, target):
        self.history.append(Move(self.white, self[source], source, target))
        self[target] = self[source]
        self[source] = PieceType.EMPTY
        self.white = not self.white
    
    def dummy_move0(self):
        self.move((7, 0), (7, 1))
    
    def dummy_move1(self):
        self.move((7, 1), (7, 0))
    
    def transpose(self, source, target):
        self.move((0, source), (2, source))
        self.dummy_move0()
        self.move((0, target), (1, target))
        self.dummy_move1()

        self.move((2, source), (2, target))
        self.dummy_move0()
        self.move((1, target), (1, source))
        self.dummy_move1()

        self.move((2, target), (0, target))
        self.dummy_move0()
        self.move((1, source), (0, source))
        self.dummy_move1()

    def get_history(self):
        return [move.format(**self.fmt_options) for move in self.history]

    def get_state(self):
        return self.board
