import datetime
import os


_DEBUG = True

_current_game = '<None>'
_move_number = 0


def init_game():
    global _current_game, _move_number

    if not _DEBUG:
        return

    _current_game = datetime.datetime.now().isoformat().replace(':', '_')
    _move_number = 0



def save_move(im_analyser):
    global _move_number
    if not _DEBUG:
        return

    _move_number += 1
    im_analyser.save_state(os.path.join('games', _current_game), str(_move_number))

