import os.path

import stockfish


ELO_RATING: int | None = None
"""
Constant denoting the ELO rating of the engine at chess.
Setting this to None defines the ELO rating to stockfish's 
default ELO rating.

- an ELO rating of around 500 denotes a beginner
- an ELO rating of 1000-1500 denotes an intermediate player
- an ELO rating of 1500-2000 denotes an expert player
- an ELO rating of above 2000 denotes a Chess Master, i.e. usually a professional player.

The highest ELO score ever obtained by a human being during official competition at the time
of writing this text is 2882 ELO by the Norwegian player Magnus Carlsen. 
(https://en.wikipedia.org/wiki/List_of_chess_players_by_peak_FIDE_rating)
"""


class StockfishEngine(stockfish.Stockfish):
    def __init__(self):
        local_directory = os.path.split(__file__)[0]
        super().__init__(os.path.join(local_directory, 'stockfish-windows-x86-64-avx2.exe'))
        if ELO_RATING is not None:
            self.set_elo_rating(ELO_RATING)

