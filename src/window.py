import sys
import tkinter
import typing_extensions
from tkinter import filedialog
from typing import Callable

from docutils.nodes import comment

from board_display import BoardDisplay
import chess
import errors
import os

from image import Image

BGCOLOR = ''
BOARD_COLUMNS = 'abcdefgh'
EMPTY_PIECE = '-'


img_files = ('png', 'jpg', 'jpeg', 'bmp')

imgfile_types = (
    ('Image file', [*(x.upper() for x in img_files), *img_files]),
)

AT = typing_extensions.ParamSpec('AT')


class Window(tkinter.Tk):
    """
    The main window of the program.

    This window currently contains one left side with the board display and one right side
    with different sections. The first section is a message to communicate to the
    user, the second is an interface for the user to be able to enter a move, the third is
    and error message and the last is a group of buttons.
    """
    def __init__(self, play_move_cb: Callable[[], None]):
        super().__init__('Application')

        self.config()

        self._board_display = BoardDisplay(self, EMPTY_PIECE)
        self._board_display.grid(row=0, column=0)
        """self._piece_color_display = BoardDisplay(self, EMPTY_PIECE)
        self._piece_color_display.grid(row=1, column=0)"""

        self._right_frame = tkinter.Frame(self, bg='green')
        self._right_frame.grid(row=0, column=1, padx=15)

        self._message_text = tkinter.StringVar(self, value='this is a message for the user.')
        self._message_lbl = tkinter.Label(self._right_frame, bg='green', textvariable=self._message_text)
        self._message_lbl.grid(row=0, column=0)

        self._input_frame = tkinter.Frame(self._right_frame, bg='green')
        # self._input_frame.grid(row=1, column=0)

        self._source_square_input = tkinter.Entry(self._input_frame, width=10)
        self._source_square_input.grid(row=0, column=0, padx=5)

        self._dest_square_input = tkinter.Entry(self._input_frame, width=10)
        self._dest_square_input.grid(row=0, column=1, padx=5)

        self._error_var = tkinter.StringVar(self, '')
        self._error_lbl = tkinter.Label(self._right_frame, fg='red', bg='green', textvariable=self._error_var)
        self._error_lbl.grid(row=2, column=0)

        self._buttons_frame = tkinter.Frame(self._right_frame, bg='green')
        self._buttons_frame.grid(row=3, column=0)

        self._single_button_frame = tkinter.Frame(self._buttons_frame, bg='green')
        # do not show yet

        self._turn_button_text = tkinter.StringVar(self, 'Turn')
        self._turn_button_cb = lambda: None
        self._turn_button = tkinter.Button(
            self._single_button_frame, bg='yellow', textvariable=self._turn_button_text, command=lambda: self._turn_button_cb()
        )
        self._turn_button.pack(pady=15)

        self._double_button_frame = tkinter.Frame(self._buttons_frame, bg='green')
        # do not show yet

        self._yesnobuttons_cb = lambda val: None
        self._yesbutton = tkinter.Button(self._double_button_frame, text='Yes', bg='yellow', command=lambda: self._yesnobuttons_cb(True))
        self._yesbutton.pack(pady=15)
        self._nobutton = tkinter.Button(self._double_button_frame, text='No', bg='yellow', command=lambda: self._yesnobuttons_cb(False))
        self._nobutton.pack(pady=15)

        self._play_move_cb = play_move_cb

        self.geometry("720x480")

        self._square_from = ''
        self._square_to = ''
        self._exists = False
        self._click_args = None
        self.withdraw()

        self._setup_starting_position()

    def ask_for_user_image(self):
        """
        Ask the user to open an image file.
        Returns the image that the user opened.
        """
        if self._exists:
            img_path = filedialog.Open(self, filetypes=imgfile_types).show()
        else:
            img_path = filedialog.askopenfilename(filetypes=imgfile_types)

        if not os.path.exists(img_path):
            raise errors.InvalidFileError("Please select an existing file.")
        name, ext = os.path.splitext(img_path)
        if ext.lower().removeprefix('.') not in img_files:
            raise errors.InvalidFileError("The specified path should be an image file.")

        img = Image.read(img_path)

        if (not any(img.shape)) or (not all(img.shape)):
            raise errors.InvalidFileError("Invalid data inside file, is it corrupted ?")

        return img

    def _setup_starting_position(self):
        board = chess.Board()
        self.update_board_display(board)

    def _on_play_move(self):
        """
        Called when the 'Play move' button is clicked.
        """
        self._square_from = self._source_square_input.get()
        self._square_to = self._dest_square_input.get()
        self._play_move_cb()

    def update_board_display(self, board):
        """
        Update the board display to the specified state of the game.
        """
        symbols = [[None for _ in range(8)] for _ in range(8)]
        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            piece = board.piece_at(square)
            symbol = EMPTY_PIECE if piece is None else piece.symbol()

            symbols[8 - rank - 1][file] = symbol

        for i in range(8):
            self._board_display.insert('', i, tags=symbols[i], values=[symbols[i][j] for j in range(8)])

    def clear_inputs(self):
        """
        Clear our text input fields.
        """
        self._source_square_input.delete(0, 'end')
        self._dest_square_input.delete(0, 'end')
        self._error_var.set('')

    def show(self):
        """
        Make the main window visible.
        """
        self.deiconify()
        self._exists = True
        self.mainloop()
        self._exists = False

    @property
    def square_from(self):
        """
        Value of the input field for the move's source square.
        """
        return self._source_square_input.get()

    @property
    def square_to(self):
        """
        Value of the input field for the move's destination square.
        """
        return self._dest_square_input.get()

    @property
    def exists(self):
        """
        True if the window is currently visible on the screen.
        """
        return self._exists

    def input_error(self, err):
        """
        Show the specified error message next to the text input fields
        for move selection.
        """
        self._error_var.set(err)

    def _make_click_cb(self, valfunc: Callable[AT, tuple]) -> Callable[AT, None]:
        def _inner(*args):
            self._click_args = valfunc(*args)

        return _inner

    def message(self, msg='', /):
        """
        Set the text message that is currently being displayed to the user.
        """
        self._message_text.set(msg)

    def hidemoveinputs(self):
        """
        Hide the input fields that let the user enter a move.
        """
        try:
            self._input_frame.grid_forget()
        except:
            pass

    def showmoveinputs(self):
        """
        Show the input fields that let the user enter a move.
        """
        try:
            self._input_frame.grid(row=1, column=0)
        except:
            pass

    def nobuttons(self):
        """
        Hide all buttons in the button group.
        """
        try:
            self._single_button_frame.grid_forget()
        except:
            pass

        try:
            self._double_button_frame.grid_forget()
        except:
            pass

        self._click_args = None

    def yesnobuttons(self, click_cb: Callable[[bool], None]):
        """
        Show the user a 'Yes' and a 'No' button, of which the command is defined by `click_cb`.
        The argument passed to `click_cb` is True when the button clicked was 'Yes', and False
        otherwise.
        """
        self.nobuttons()

        self._yesnobuttons_cb = click_cb

        self._double_button_frame.grid(row=4, column=0)

    def singlebutton(self, btn_text: str, click_cb: Callable[[], None]):
        """
        Show the user
        """
        self.nobuttons()

        self._turn_button_text.set(btn_text)
        self._turn_button_cb = click_cb

        self._single_button_frame.grid(row=4, column=0)

    def screen_before_start(self):
        """
        Show the starting screen to the user.

        The return value of `wait_for_click` is the empty tuple.
        """
        """def _inner():
            self._click_args = ()"""
        _inner = self._make_click_cb(lambda: ())

        self.message("Click 'Start' when you are ready. Don't forget to turn on the spotlight and \nto place the pieces in the starting position.")
        self.hidemoveinputs()
        self.singlebutton('Start', _inner)

    def screen_wait_turn(self):
        """
        Show the "Waiting for user to play" screen to the user.

        The return value of `wait_for_click` is the empty tuple.
        """
        """def _inner():
            self._click_args = ()"""
        _inner = self._make_click_cb(lambda: ())

        self.message("Please click 'Turn' once you have played your move.")
        self.hidemoveinputs()
        self.singlebutton('Turn', _inner)

    def screen_validate_move(self, detected_move: str):
        """
        Ask the user to validate the move that was detected.

        The return value of `wait_for_click` is `(is_yes,)`, where
        `is_yes` is True if the 'Yes' button was clicked and False otherwise.
        """
        """def _inner(val):
            self._click_args = (val,)"""
        _inner = self._make_click_cb(lambda val: (val,))

        self.message(f"The move {detected_move!r} was detected. Is this correct ?")
        self.hidemoveinputs()
        self.yesnobuttons(_inner)

    def screen_fix_move(self):
        """
        Ask the user what move they have played.

        The return value of `wait_for_click` is a `(_from, _to)` tuple
        of string representations of the source and destination squares of the
        move provided by the user.
        """
        """def _inner():
            self._click_args = (self.square_from, self.square_to)"""

        _inner = self._make_click_cb(lambda: (self.square_from, self.square_to))

        self.message("Please type in the correct move and click 'Confirm'.")
        self.showmoveinputs()
        self.singlebutton('Confirm', _inner)

    def screen_thinking(self):
        """
        Show the user that the engine is thinking.

        `wait_for_click` waits forever because this screen has no buttons.
        """
        self.message("Game engine is thinking...")
        self.hidemoveinputs()
        self.nobuttons()

    def screen_wait_robot_move(self, selected_move: str):
        """
        Show the user that we are waiting for the robot's move to be played.

        The return value of `wait_for_click` is the empty tuple.
        """
        """def _inner():
            self._click_args = ()"""
        _inner = self._make_click_cb(lambda: ())

        self.message(f"The robot chose move {selected_move!r}. Please click 'Confirm' once that move has been played.")
        self.hidemoveinputs()
        self.singlebutton('Confirm', _inner)

    def screen_robot_moving(self):
        """
        Show the user that we are waiting for the robotic arm to
        finish its current task.

        `wait_for_click` never returns because this screen has no buttons.
        """
        self.message("Operating the move...")
        self.hidemoveinputs()
        self.nobuttons()

    def screen_robot_calibrating(self):
        self.message("Calibrating the robotic arm, please wait...")
        self.hidemoveinputs()
        self.nobuttons()

    def screen_loading(self):
        """
        Show the user that we are currently performing some costly operations.

        `wait_for_click` never returns because this screen has no buttons.
        """
        self.message("Loading, please wait...")
        self.hidemoveinputs()
        self.nobuttons()

    def wait_for_click(self):
        """
        Wait for the user to click one of the buttons in the currently visible screen.
        Returns a tuple of values that depends on what that screen is.
        """
        while self._click_args is None:
            if not self.exists:
                sys.exit(0)

        result = self._click_args
        self._click_args = None
        return result

