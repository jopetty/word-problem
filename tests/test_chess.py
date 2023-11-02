import unittest
import pyrootutils
from state_tracking.chess import BoardTracker
from state_tracking.chess.piece_type import PieceType


PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

class TestData(unittest.TestCase):  # noqa: D101

    def test_chess(self):
        board = BoardTracker.queen_rook_permutations()
        board.transpose(0, 1)
        board.transpose(3, 4)
        board.transpose(2, 1)
        self.assertListEqual(
            board.board[0, :5].tolist(),
            [PieceType.WHITE_QUEEN.value, PieceType.WHITE_QUEEN.value, PieceType.WHITE_ROOK.value, PieceType.WHITE_QUEEN.value, PieceType.WHITE_QUEEN.value],
        )
        self.assertListEqual(
            [m.format() for m in board.history],
            ['Ra8c8', 'Rh8h7', 'Qa7b7', 'Rh7h8', 'Rc8c7', 'Rh8h7', 'Qb7b8', 'Rh7h8', 'Rc7a7', 'Rh8h7', 'Qb8a8', 'Rh7h8', 'Qa5c5', 'Rh8h7', 'Qa4b4', 'Rh7h8', 'Qc5c4', 'Rh8h7', 'Qb4b5', 'Rh7h8', 'Qc4a4', 'Rh8h7', 'Qb5a5', 'Rh7h8', 'Qa6c6', 'Rh8h7', 'Ra7b7', 'Rh7h8', 'Qc6c7', 'Rh8h7', 'Rb7b6', 'Rh7h8', 'Qc7a7', 'Rh8h7', 'Rb6a6', 'Rh7h8'],
        )
