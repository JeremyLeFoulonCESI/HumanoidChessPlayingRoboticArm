import sys
import time

import chess

import cv2

from color_board import ChessColor
from window import Window
from stockfish_implementation import StockfishEngine
from image_analysis import PositionAnalyser, _sq_chessmodule
import debug_info
from robot import RoboticArm


NO_ENGINE = False  # False if the system is to use the stockfish engine
NO_ARM = True  # False if the mechanical arm is connected by USB to the current machine
NO_CAM = False  # False if the camera is directly connected by USB to the current machine
NO_PHYSICAL_BUTTONS = True  # False if physical buttons are connected to the current machine


def _square_from_chessmodule(sq):
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)

    return _sq_chessmodule(file, rank)


class ChessGame:
    def __init__(self, window: Window, chess_computer: StockfishEngine):
        self._turn = ChessColor.WHITE
        self._computer = chess_computer
        self._window = window
        self._board = chess.Board()
        self._winner = None
        self._finished = False
        self._ui_played = False
        self._ui_played_move = None

        # if not NO_ARM:
        """self._robotic_arm = RoboticArm('COM3')
        time.sleep(1)
        self._robotic_arm.calibrate()
        self._robotic_arm.rest_position()"""

        self._pos_analyser = PositionAnalyser(
            self._window,
            'USB Camera',  # use the display name of our camera
            cv2.CAP_DSHOW,
            askopenfilename=NO_CAM,
            cap_exposure=-8,
            cap_framecount=10,
            cap_delay_ms=2500,
        )
        """print('Please turn on the spot light to avoid shadows casting on the board.')
        print('Press enter when that is done.')
        input()"""

    def validate_move(self, move):
        return self._board.is_legal(move)

    @classmethod
    def wait_user_input(cls, msg, prompt='', /):
        print(msg)
        if NO_PHYSICAL_BUTTONS:
            input(prompt)
        else:
            print('(Press the button to continue)')
            # wait for user to press the button

    def _require_fix_move(self):
        self._window.screen_fix_move()

        while True:
            _from, _to = self._window.wait_for_click()
            self._window.clear_inputs()
            for sq in (_from, _to):
                if (
                        (len(sq) != 2) or
                        (sq[0] not in "abcdefgh") or
                        (not sq[1].isdigit()) or
                        (int(sq[1]) not in range(1, 9))
                ):
                    self._window.input_error('Invalid input. Please try again.')
                    continue

            try:
                move = self._board.find_move(chess.parse_square(_from), chess.parse_square(_to))
            except chess.IllegalMoveError:
                self._window.input_error('Illegal move. Please try again.')
                continue
            return move

    def user_play(self):

        self._window.screen_wait_turn()
        self._window.wait_for_click()  # wait for the user to click the 'Turn' button

        self._window.screen_loading()

        # take an image of the position
        if self._pos_analyser.push(self._window, askopenfilename=NO_CAM) is None:
            move = None

        else:
            # detect the piece that was moved
            move = self._pos_analyser.guess_move(self._board, self._turn)

        if move is None:  # this is True when we know we failed to detect the move
            move = self._require_fix_move()
        else:
            self._window.screen_validate_move(move.uci())
            if not self._window.wait_for_click()[0]:  # the user said that the move is incorrect
                move = self._require_fix_move()

        return move

    def ui_play(self, move):

        if self._board.is_legal(move):
            _from = 63 - move.from_square
            _to = 63 - move.to_square

            self._ui_played_move = move
            self._ui_played = True
            return True
        return False

    def wait_for_ui_play(self):
        while True:
            if self._ui_played:
                self._ui_played = False
                if self.validate_move(self._ui_played_move):
                    self._window.input_error('')
                    break
                self._window.input_error("Illegal move!")

            if not self._window.exists:
                sys.exit(0)
        self._ui_played = False


        print(f"Robot is playing {self._ui_played_move.uci()}.")
        print("Press enter once the move has been played.")
        input()

        self._pos_analyser.push(self._window, askopenfilename=NO_CAM)
        return self._ui_played_move

    def robot_play(self):
        if NO_ENGINE:
            return self.wait_for_ui_play()

        move_str = self._computer.get_best_move_time(500)
        if move_str is None:
            return None

        move = chess.Move.from_uci(move_str)

        """print(f"Robot is playing {move.uci()}.")
        print("Press enter once the robot is done.")
        input()"""

        if NO_ARM:
            self._window.screen_wait_robot_move(move.uci())
            self._window.wait_for_click()
        else:
            self._window.screen_robot_moving()
            ep_square = _square_from_chessmodule(self._board.ep_square) if self._board.is_en_passant(move) else None
            castling_side = self._board.is_queenside_castling(move) if self._board.is_castling(move) else None

            self._robotic_arm.make_move(  # make the robot perform the move - this is a blocking operation
                _square_from_chessmodule(move.from_square), _square_from_chessmodule(move.to_square),
                self._board.piece_at(move.from_square), self._board.piece_at(move.to_square),
                castling_side=castling_side, en_passant_square=ep_square
            )

        self._window.screen_loading()

        self._pos_analyser.push(self._window, askopenfilename=NO_CAM)

        return move

    def half_turn(self):

        if self._turn == self._pos_analyser.robot_color:
            move = self.robot_play()

        else:
            move = self.user_play()

        self._computer.make_moves_from_current_position([move.uci()])
        self._board.push(move)
        self._window.update_board_display(self._board)

        if self._board.is_checkmate():
            self._winner = self._turn
            return True

        return self._board.is_stalemate() or self._board.is_repetition() or self._board.is_insufficient_material()

    def turn(self):
        self._turn = ChessColor.WHITE

        if self.half_turn():
            return True

        self._turn = ChessColor.BLACK

        return self.half_turn()

    def start(self):
        while not self._window.exists:
            pass

        self._window.screen_before_start()
        self._window.wait_for_click()  # wait for the user to click the button

        """if not NO_ARM:
            self._window.screen_robot_calibrating()
            self._robotic_arm.calibrate()"""

        self._window.screen_loading()
        self._pos_analyser.start(self._window, askopenfilename=NO_CAM)

        debug_info.init_game()
        while not self.turn():
            if not self._window.exists:
                break

        self._finished = True

