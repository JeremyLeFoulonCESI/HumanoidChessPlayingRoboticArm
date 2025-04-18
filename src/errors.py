

class AppError(Exception):
    pass


class InvalidChessboardError(AppError):
    pass


class ChessboardNotFoundError(AppError):
    pass


class InvalidFileError(AppError):
    pass


class CameraNotFoundError(AppError):
    pass

