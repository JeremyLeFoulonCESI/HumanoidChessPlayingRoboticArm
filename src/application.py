import sys
import threading
import chess
import chess_game

import window
from stockfish_implementation import StockfishEngine


class Application:
    def __init__(self):
        self.window = window.Window(self.on_play_move)  # this arg is temporary
        self.stockfish = StockfishEngine()
        self.finished = False
        self.game = chess_game.ChessGame(self.window, self.stockfish)

    def _start(self):
        try:
            self.game.start()
        except SystemExit:
            pass
        finally:
            try:
                self.window.destroy()
            except:
                pass

    @classmethod
    def run(cls):
        self = cls()

        # we run the window's display loop on the main thread
        # and the chess game loop on a separate thread
        # so that the chess game loop can perform long computations
        # without lagging the window's visual updating.

        # start a thread for managing the game state
        th = threading.Thread(target=self._start)
        th.start()

        # tkinter IS REQUIRED to run on the main thread
        # because it is not thread-aware.
        self.window.show()
        th.join()

    def on_play_move(self):
        if self.finished:
            return

        if self.window.square_from not in chess.SQUARE_NAMES:
            return

        if self.window.square_to not in chess.SQUARE_NAMES:
            return

        try:
            _from = chess.SQUARE_NAMES.index(self.window.square_from)
            _to = chess.SQUARE_NAMES.index(self.window.square_to)
            move = chess.Move(_from, _to)
        except:
            self.window.clear_inputs()
            sys.excepthook(*sys.exc_info())
        else:
            self.window.clear_inputs()
            if not self.game.ui_play(move):
                print(f'Illegal move {move} (from {_from} to {_to}) by user', file=sys.stderr)

