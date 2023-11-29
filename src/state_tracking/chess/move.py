from typing import NamedTuple

from .piece_type import PieceType

LETTERS = "abcedfgh"


def format_position(position):
    i, j = position
    return f"{LETTERS[i]}{8 - j}"


class Move(NamedTuple):
    white: bool
    piece_type: PieceType
    source: tuple
    target: tuple

    def __repr__(self):
        return self.format()

    def format(self, piece_type=True, source=True, target=True) -> str:
        tokens = []
        if piece_type:
            tokens.append(self.piece_type.get_code())
        if source:
            tokens.append(format_position(self.source))
        if target:
            tokens.append(format_position(self.target))
        return " ".join(tokens)
