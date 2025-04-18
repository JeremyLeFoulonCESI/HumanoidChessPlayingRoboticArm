import tkinter
from tkinter import ttk


BOARD_COLUMNS = 'abcdefgh'


class BoardDisplay(ttk.Treeview):
    """
    Tkinter widget that can display the state of a chessboard on the screen.
    This is currently a "spread sheet", but could be changed to something else
    to make it look better.
    """
    def __init__(self, master, starting_value=''):
        super().__init__(
            master,
            columns=[str(i) for i in range(8)],
            height=8,
            selectmode='none',
        )
        self.column('#0', width=0, stretch=tkinter.NO)
        for i in range(8):
            self.column(str(i), width=30, anchor=tkinter.CENTER)
            self.heading(str(i), text=BOARD_COLUMNS[i])
            self.insert('', i, values=[starting_value for _ in range(8)], tags=[str(i) for i in range(8)])

    def set_cells_at(self, rank, **file_values):
        self.insert('', 8 - rank, tags=[*file_values.keys()], values=[*file_values.values()])

    def set_cell_at(self, rank, file, value):
        return self.set_cells_at(rank, **{file: value})

