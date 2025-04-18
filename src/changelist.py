from collections import namedtuple
from collections.abc import Callable
from color_board import ChessColor

import chess
import numpy



# describes which squares are valid for castling.
_CASTLING_SQUARES = [
    {  # black
        'q': {  # queen side castle
            (4, 0): chess.KING,  # e8  (src square)
            (2, 0): None,        # c8  (dst square)
            (0, 0): chess.ROOK,  # a8
            (3, 0): None,        # d8
        },
        'k': {  # king side castling
            (4, 0): chess.KING,  # e8  (src square)
            (6, 0): None,        # g8  (dst square)
            (7, 0): chess.ROOK,   # h8
            (5, 0): None,        # f8
        },
        'both': (4, 0),  # e8
    },
    {  # white
        'q': {  # queen side castle
            (4, 7): chess.KING,  # e1  (src square)
            (2, 7): None,        # c1  (dst square)
            (0, 7): chess.ROOK,  # a1
            (3, 7): None,        # d1
        },
        'k': {  # king side castling
            (4, 7): chess.KING,  # e1  (src square)
            (6, 7): None,        # g1  (dst square)
            (7, 7): chess.ROOK,   # h1
            (5, 7): None,        # f1
        },
        'both': (4, 7),  # e1
    },
]


def _sq_chessmodule(x, y):
    return x, 7 - y


_ChangeItem = namedtuple('_ChangeItem', ['value', 'x', 'y'])

class ChangeList:
    """
    A type that stores changes between two positions using a bitboard.
    """

    def __init__(self):
        self._bitboard = 0

    def _set(self, mask):
        self._bitboard |= mask

    def _unset(self, mask):
        self._bitboard &= ~mask

    def _get(self, mask):
        return self._bitboard & mask

    @classmethod
    def _group_mask(cls, *coords):
        mask = 0
        for x, y in coords:
            mask |= cls._coords_mask(x, y)
        return mask

    @staticmethod
    def _coords_mask(x, y):
        index = 8 * y + x
        return 1 << index

    def group_set(self, *coords: tuple[int, int]):
        mask = self._group_mask(*coords)
        return self._set(mask)

    def group_unset(self, *coords: tuple[int, int]):
        mask = self._group_mask(*coords)
        return self._unset(mask)

    def group_get(self, *coords: tuple[int, int]) -> int:
        mask = self._group_mask(*coords)
        return self._get(mask)

    def set(self, x: int, y: int):
        mask = self._coords_mask(x, y)
        return self._set(mask)

    def unset(self, x: int, y: int):
        mask = self._coords_mask(x, y)
        return self._unset(mask)

    def get(self, x: int, y: int) -> bool:
        mask = self._coords_mask(x, y)
        return self._get(mask) > 0

    def clear(self):
        self._bitboard = 0

    def copy(self):
        result = ChangeList()
        result._bitboard = self._bitboard
        return result

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        return self.copy()

    def __getitem__(self, item: tuple[int, int]) -> bool:
        x, y = item
        return self.get(x, y)

    def __setitem__(self, key: tuple[int, int], value: bool):
        x, y = key
        if value:
            return self.set(x, y)
        return self.unset(x, y)

    def __iter__(self):
        return self.where(lambda a: True)

    def where(self, cb: Callable[[bool], bool]):
        for x, y in numpy.ndindex((8, 8)):
            val = self.get(x, y)
            if cb(val):
                yield _ChangeItem(val, x, y)

    def where_true(self):
        return self.where(lambda a: a)

    def where_false(self):
        return self.where(lambda a: not a)

    @staticmethod
    def _guess_simple_move(changes: list[_ChangeItem], prev_board: chess.Board, turn):
        idx_from = None

        for i, change in enumerate(changes):
            _x, _y = _sq_chessmodule(change.x, change.y)
            piece = prev_board.piece_at(chess.square(_x, _y))
            if (piece is not None) and (piece.color == turn):
                idx_from = i
                break

        if idx_from is None:
            return None, None

        _change_from = changes[idx_from]
        _change_to = changes[not idx_from]

        return (_change_from.x, _change_from.y), (_change_to.x, _change_to.y)

    @staticmethod
    def _guess_en_passant(changes, prev_board, turn):
        idx_from = None
        idx_to = None

        for i, change in enumerate(changes):
            _x, _y = _sq_chessmodule(change.x, change.y)
            piece = prev_board.piece_at(chess.square(_y, _x))
            if piece is None:
                idx_to = i
                continue
            if piece.color == turn:
                idx_from = i

        if None in (idx_to, idx_from):
            return None, None

        _change_from = changes[idx_from]
        _change_to = changes[idx_to]

        return (_change_from.x, _change_from.y), (_change_to.x, _change_to.y)

    @staticmethod
    def _guess_castling(changes, prev_board, turn):
        castling_squares = _CASTLING_SQUARES[turn]

        side = None

        for i, change in enumerate(changes):
            _x, _y = _sq_chessmodule(change.x, change.y)

            sq = (change.x, change.y)
            if sq == castling_squares['both']:
                continue
            if sq in castling_squares['q']:
                _side = 'q'
            elif sq in castling_squares['k']:
                _side = 'k'
            else:
                return None, None  # not a castling square

            if side is None:
                side = _side
            elif _side != side:
                return None, None  # not all squares represent the same castling side

            piece = prev_board.piece_at(chess.square(_x, _y))

            piece_color = None if piece is None else piece.color

            if (piece_color is not None) and (piece_color != turn):
                return None, None  # wrong piece color was on that square before

            piece_type = None if piece is None else piece.piece_type

            if piece_type != castling_squares[_side][sq]:
                return None, None   # wrong piece type was on that square before

        _from, _to, *_ = castling_squares[side].keys()
        return _from, _to

    @classmethod
    def _guess_complex_move(cls, changes, prev_board, turn):
        if len(changes) == 3:  # can only be an en passant
            return cls._guess_en_passant(changes, prev_board, turn)
        if len(changes) == 4:
            return cls._guess_castling(changes, prev_board, turn)
        return None, None

    def guess_move(self, prev_board: chess.Board, turn: ChessColor) -> tuple[tuple[int, int], tuple[int, int]] | tuple[None, None]:
        changes = [*self.where_true()]

        turn = turn.value == 'w'

        if len(changes) == 2:
            return self._guess_simple_move(changes, prev_board, turn)
        if len(changes) > 2:
            return self._guess_complex_move(changes, prev_board, turn)
        return None, None

