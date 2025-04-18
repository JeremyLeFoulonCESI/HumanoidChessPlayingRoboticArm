"""
Color boards. Apart from ChessColor, this is not used by the rest of the code right now.
"""


from enum import Enum, IntEnum


class ChessColor(Enum):
    WHITE = 'w'
    BLACK = 'b'


class SquareChangeType(IntEnum):
    CHANGE_COLOR = 0
    DISAPPEAR = 1
    APPEAR = 2


STARTING_COLOR_FEN = 'wwwwwwww/wwwwwwww/8/8/8/8/bbbbbbbb/bbbbbbbb'
EMPTY_FEN = '8/8/8/8/8/8/8/8'

file_names = 'abcdefgh'


def square_file(sq):
    return sq % 8


def square_rank(sq):
    return 8 - (sq // 8)


def square_from_pair(file, rank):
    return 8 * rank + file


def square_from_str(s):
    s_file, s_rank = s
    file = file_names.index(s_file)
    rank = int(s_rank)
    return square_from_pair(file, 8 - rank)


def square_name(s):
    if s is None:
        return None
    rank = square_rank(s)
    file = square_file(s)

    return f"{file_names[file]}{rank}"


class ColorBoardDifference:
    def __init__(self, bits, before, after):
        self._bits = bits
        self._before = before
        self._after = after

    def changed_squares(self):
        result = []

        for i in range(64):
            if (self._bits >> i) & 1:
                result.append(i)
        return result

    def change_type(self):
        """
        Describes how the board has changed between before and after
        Yields: tuple[SquareChangeType, Literal['w', 'b'] | None, Literal['w', 'b'] | None]
        """
        for sq in self.changed_squares():
            (cbefore,) = self._before.get_squares((sq,))
            (cafter,) = self._after.get_squares((sq,))
            if cbefore is not None:
                if cafter is None:
                    yield SquareChangeType.DISAPPEAR, sq, cbefore, cafter
                    continue
                yield SquareChangeType.CHANGE_COLOR, sq, cbefore, cafter
            else:
                yield SquareChangeType.APPEAR, sq, cbefore, cafter

    @staticmethod
    def _simple_played_move(changes):
        _from = (None, None)
        _to = (None, None)

        for kind, sq, before, after in changes:
            if kind == SquareChangeType.DISAPPEAR:
                _from = (sq, before)
            elif (after == _from[1]) or (_from[1] is None):
                _to = (sq, after)

        return _from[0], _to[0]

    @staticmethod
    def _special_played_move(changes):
        disappear = []
        appear = []
        change_color = []

        for kind, sq, before, after in changes:
            if kind == SquareChangeType.CHANGE_COLOR:
                change_color.append((sq, before, after))
            elif kind == SquareChangeType.DISAPPEAR:
                disappear.append((sq, before, after))
            else:
                appear.append((sq, before, after))

        if (len(disappear), len(appear)) == (2, 1) and len(change_color) == 0:  # en passant
            print('en passant')
            disappear_ranks = [square_rank(dis[0]) for dis in disappear]
            disappear_files = [square_file(dis[0]) for dis in disappear]

            # the two pieces that have disappeared must be next to each other, on the same rank
            if abs(disappear_files[0] - disappear_files[1]) != 1 or (disappear_ranks[0] != disappear_ranks[1]):
                return None, None

            appear_files = [square_file(appe[0]) for appe in appear]
            idx = disappear_files.index(appear_files[0])

            # one of the disappeared pieces must be on the same file and next to the appeared piece
            if idx < 0:
                return None, None
            appear_ranks = [square_rank(appe[0]) for appe in appear]
            if abs(disappear_ranks[idx] - appear_ranks[0]) != 1:
                return None, None

            if (appear[0][2] == 'w') and (appear_ranks[0] >= disappear_ranks[idx]):
                return None, None
            elif (appear[0][2] == 'b') and (appear_ranks[0] <= disappear_ranks[idx]):
                return None, None

            return disappear[not idx][0], appear[0][0]


        if (len(disappear) == len(appear) == 2) and len(change_color) == 0:  # castling
            color = None

            print('castling')

            for sq, before, after in appear:
                if color is None:
                    color = after
                elif after != color:
                    return None, None

            print('appear colors are the same')

            for sq, before, after in disappear:
                if before != color:
                    return None, None

            print('all colors are the same', color)

            dis_squares = [dis[0] for dis in disappear]
            app_squares = [app[0] for app in appear]

            for s in ('a1', 'e1', 'c1', 'd1'):
                _is = square_from_str(s)
                print(f"square name={s}, square num={_is}")

            if color == 'w':
                if (
                    (square_from_str('a1') in dis_squares) and (square_from_str('e1') in dis_squares) and
                    (square_from_str('c1') in app_squares) and (square_from_str('d1') in app_squares)
                ):
                    return square_from_str('e1'), square_from_str('c1')
                if (
                    (square_from_str('h1') in dis_squares) and (square_from_str('e1') in dis_squares) and
                    (square_from_str('g1') in app_squares) and (square_from_str('f1') in app_squares)
                ):
                    return square_from_str('e1'), square_from_str('g1')
            elif color == 'b':
                if (
                        (square_from_str('a8') in dis_squares) and (square_from_str('e8') in dis_squares) and
                        (square_from_str('c8') in app_squares) and (square_from_str('d8') in app_squares)
                ):
                    return square_from_str('e8'), square_from_str('c8')
                if (
                        (square_from_str('h8') in dis_squares) and (square_from_str('e8') in dis_squares) and
                        (square_from_str('g8') in app_squares) and (square_from_str('f8') in app_squares)
                ):
                    return square_from_str('e8'), square_from_str('g8')
            else:
                return None, None

        print(len(disappear), len(appear))
        return None, None

    def find_played_move(self):
        changes = tuple(self.change_type())

        if len(changes) == 2:
            return self._simple_played_move(changes)
        return self._special_played_move(changes)

    def __str__(self):
        result = "Color board difference:\n"

        for i in range(8):
            for j in range(8):
                pos = i * 8 + j
                val = bool(self._bits & (1 << pos))
                result += str(int(val))
                if j < 7:
                    result += ' '
            if i < 7:
                result += '\n'

        return result


class ColorBoard:
    def __init__(self):
        self._occupancy_bitboard = 0
        self._color_bitboard = 0

    def set_squares(self, square_to_value):
        for sq, val in square_to_value.items():
            (sq,) = self._cells(sq)
            if val is None:
                # print('here', sq)
                self._clear_occupancy(1 << sq)
                self._clear_color(1 << sq)
                continue
            self._set_occupancy(1 << sq)
            if val in (ChessColor.WHITE, 'w'):
                self._set_color(1 << sq)
                continue
            self._clear_color(1 << sq)

    def get_square(self, sq):
        gen = self.get_squares([sq])
        return next(gen, None)

    def set_square(self, sq, val):
        sqval = {sq: val}
        self.set_squares(sqval)

    def flip(self, inplace=True):
        result = self if inplace else ColorBoard()
        for first in range(32):
            second = 64 - first - 1
            tmp = self.get_square(first)
            result.set_square(first, self.get_square(second))
            result.set_square(second, tmp)
        return result

    def get_squares(self, squares):
        for sq in self._cells(*squares):
            if not self._occupancy_bitboard & (1 << sq):
                yield None
                continue
            if self._color_bitboard & (1 << sq):
                yield ChessColor.WHITE
                continue
            yield ChessColor.BLACK

    def move_piece(self, sq_from, sq_to):
        p_src = self.get_square(sq_from)
        self.set_squares({sq_from: None, sq_to: p_src})

    def _get_squares(self, mask):
        return self._occupancy_bitboard & mask, self._color_bitboard & mask

    @staticmethod
    def _cells(*squares):
        for sq in squares:
            if isinstance(sq, str):
                str_file, str_rank = sq
                file = file_names.index(str_file)
                rank = 8 - int(str_rank)
                yield 8 * rank + file
            else:
                yield sq

    def _set_occupancy(self, mask):
        self._clear_occupancy(mask)
        self._occupancy_bitboard |= mask

    def _set_color(self, mask):
        self._clear_color(mask)
        self._color_bitboard |= mask

    def _clear_occupancy(self, mask):
        self._occupancy_bitboard = self._occupancy_bitboard & ~mask

    def _clear_color(self, mask):
        self._color_bitboard = self._color_bitboard & ~mask

    def difference(self, other):
        """
        Compute the difference between two color boards, where other is the
        previous position and self is the next position.
        """
        return ColorBoardDifference(
            (self._occupancy_bitboard ^ other._occupancy_bitboard) |
            (self._color_bitboard ^ other._color_bitboard),
            other, self
        )

    def __copy__(self):
        result = type(self)()
        result._occupancy_bitboard = self._occupancy_bitboard
        result._color_bitboard = self._color_bitboard
        return result

    def copy(self):
        return self.__copy__()

    def __str__(self):
        result = "Color board:\n"

        squares = [*self.get_squares(range(64))]

        for i in range(8):
            for j in range(8):
                pos = i * 8 + j
                val = squares[pos]
                result += ('-' if val is None else val.value)
                if j < 7:
                    result += ' '
            if i < 7:
                result += '\n'

        return result

