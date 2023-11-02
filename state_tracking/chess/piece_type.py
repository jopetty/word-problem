from enum import Enum, auto

class PieceType(Enum):
    EMPTY = auto()
    WHITE_PAWN = auto()
    WHITE_ROOK = auto()
    WHITE_QUEEN = auto()
    WHITE_KING = auto()

    BLACK_ROOK = auto()
    BLACK_KING = auto()

    def get_code(self):
        if self == PieceType.EMPTY:
            return "."
        return self.name.split("_")[1][0]
