import json
import math
import os.path
import sys
import time
import chess
import numpy
import collections
import cv2
import errors
import debug_info

from color_board import ChessColor
from changelist import ChangeList
from window import Window
from image import Image
from cv2_enumerate_cameras import enumerate_cameras


SQUARE_SIZE = 50  # edge size of a square on the warped image in pixels

# multiplied by stddev to obtain threshold
_WHITE_CHANGE_FACTOR = 4.5  # 4 | 6 | 5.5 | 5 | 4
_BLACK_CHANGE_FACTOR = 4.5  # 8 | 6 | 5.5 | 4

# section of a chess square where the pixels have lower weights when averaging square colors
_PIXEL_WEIGHT_RATE = 0.1

# how to rotate the image to account for the orientation of the camera, or None is no rotation is needed.
_IMG_ROTATE_FIX = None  # should be None or an opencv ROTATE_* constant


def _find_cam_id_by_name(name: str) -> int:  # NOTE: only tested on Windows
    """
    Camera names have the format "<Display name> (<hex ID>)".
    This function supports full name, just display name, and just hex ID.

    If more than one camera matches the name, the index of the first camera
    to match the name is returned.

    If no cameras match the name, -1 is returned.
    """
    for cam_info in enumerate_cameras(cv2.CAP_DSHOW):
        if cam_info.name == name:
            return cam_info.index

        if '(' in cam_info.name:
            text_name, hex_id, *_ = cam_info.name.split('(')
            hex_id = hex_id.removesuffix(')')
        else:
            text_name = cam_info.name
            hex_id = ""

        text_name = text_name.strip()
        hex_id = hex_id.strip()

        if name in (text_name, hex_id):
            return cam_info.index

    return -1


def _warped_coord(x, y):
    """
    Determine the coordinates of the topleft corner of the specified square
    in the warped images.
    """
    x, y = _warped_size(x, y)
    return x + 1 * SQUARE_SIZE, y + 1 * SQUARE_SIZE

def _warped_size(w, h):
    """
    Determine the size in pixels of a rectangle, of which the dimensions are specified
    as an amount of chess squares, on the warped image
    """
    return round(w * SQUARE_SIZE), round(h * SQUARE_SIZE)


_WARPED_SQUARE_SIZE = _warped_size(1, 1)


# coordinates of the corners of the chessboard on warped images
_CORNERS_WARPED = (
    _warped_coord(1, 1),
    _warped_coord(7, 1),
    _warped_coord(1, 7),
    _warped_coord(7, 7),
)


# size of warped images in pixels
_WARPED_IMG_SIZE = _warped_size(10, 10)


_CALIBRATION_PATH = os.path.join(os.path.split(__file__)[0], 'camera_calibration.json')


def _find_center(chessboard_intersections):
    """
    Find the center of gravity of the board's intersection grid.
    This is done by averaging the x and y coordinates of all
    points of the grid
    """
    mean_point = (
        sum(x[0][0] for x in chessboard_intersections) / len(chessboard_intersections),
        sum(x[0][1] for x in chessboard_intersections) / len(chessboard_intersections),
    )

    return mean_point

_HorizontalRectangle = collections.namedtuple("_HorizontalRectangle", ("min_x", "min_y", "max_x", "max_y"))

def _find_enclosing_rectangle(chessboard_intersections):
    """
    Find the smallest horizontal rectangle that encloses the chessboard
    intersection grid
    """
    min_x = numpy.inf
    min_y = numpy.inf

    max_x = -numpy.inf
    max_y = -numpy.inf

    for array_point in tuple(chessboard_intersections):
        x, y = array_point[0]

        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return _HorizontalRectangle(min_x, min_y, max_x, max_y)


def _is_rectangle_border(x, y, rectangle, dx, dy):
    """
    Decide whether a point should be considered belonging to the enclosing rectangle
    of the intersection grid.
    """

    x_threshold = dx / 10
    y_threshold = dy / 10

    return (
        (abs(x - rectangle.min_x) < x_threshold) or
        (abs(x - rectangle.max_x) < x_threshold) or
        (abs(y - rectangle.min_y) < y_threshold) or
        (abs(y - rectangle.max_y) < y_threshold)
    )


_CornerIndices = collections.namedtuple("_CornerIndices", ("topleft", "topright", "bottomleft", "bottomright"))

def _find_corners(chessboard_intersections):
    """
    Identify the corner points of the chessboard's intersection grid.
    """

    # find the visual center of the chessboard.
    # due to the perspective, it is different from the actual center of the board.
    x0, y0 = _find_center(chessboard_intersections)

    # find the rectangle that perfectly encloses our chessboard on the image
    enclosing = _find_enclosing_rectangle(chessboard_intersections)

    topleft_index = -1
    topright_index = -1
    bottomleft_index = -1
    bottomright_index = -1

    topleft_best = -1
    topright_best = -1
    bottomleft_best = -1
    bottomright_best = -1

    # for each quarter of the visual board, the point corresponding to the
    # associated corner is the furthest away from the center that belongs to
    # the enclosing rectangle.
    for index, array_point in enumerate(tuple(chessboard_intersections)):
        x, y = array_point[0]

        dx = abs(x - x0)
        dy = abs(y - y0)

        # if a point does not belong to the enclosing rectangle,
        # it is not a corner of the intersection grid.
        if not _is_rectangle_border(x, y, enclosing, dx, dy):
            continue

        # print(dx, dy, dx+dy, index)

        # depending on what quarter of the board the point is in, we assign it to
        # the corresponding corner.

        if (x < x0) and (y < y0):
            if (dx + dy) > topleft_best:
                topleft_index = index
                topleft_best = dx + dy
        if (x > x0) and (y < y0):
            if (dx + dy) > topright_best:
                topright_index = index
                topright_best = dx + dy
        if (x < x0) and (y > y0):
            if (dx + dy) > bottomleft_best:
                bottomleft_index = index
                bottomleft_best = dx + dy
        if (x > x0) and (y > y0):
            if (dx + dy) > bottomright_best:
                bottomright_index = index
                bottomright_best = dx + dy

    return _CornerIndices(topleft_index, topright_index, bottomleft_index, bottomright_index)


def _find_chessboard_corners(img, pattern_size):
    res, corners = cv2.findChessboardCornersSB(img, pattern_size)
    if res:
        return True, corners
    return cv2.findChessboardCorners(img, pattern_size)


def _get_transformation_matrix(img_board):
    """
    Compute the transformation matrix for the image of our
    empty board. The associated transformation will warp the
    board such that all squares have an actual square shape
    and are of equal size.

    It will also give the board an overall
    offset in the image to allow visually overlapping pieces to
    remain visible in their entirety.
    """
    res, intersections = _find_chessboard_corners(img_board, (7, 7))

    cpy = img_board.copy()
    cv2.drawChessboardCorners(cpy, (7, 7), intersections, res)
    cv2.imshow('corners', cpy)
    cv2.waitKey()

    if not res:
        return None

    corner_indices = _find_corners(intersections)

    corners_sec_to_last = (
        intersections[corner_indices.topleft],
        intersections[corner_indices.topright],
        intersections[corner_indices.bottomleft],
        intersections[corner_indices.bottomright],
    )

    return cv2.getPerspectiveTransform(numpy.float32(corners_sec_to_last), numpy.float32(_CORNERS_WARPED))


def _warp_chessboard_image(img, transformation_matrix):
    """
    Warp the specified image according to the specified transformation
    matrix.
    """
    return cv2.warpPerspective(img, transformation_matrix, _WARPED_IMG_SIZE).view(type(img))


def _load_calibration(filename):
    with open(filename, mode='r') as fs:
        data = json.load(fs)

    mtx = numpy.array(data.get('matrix', []))
    dist = numpy.array(data.get('distort', []))
    rvecs = [numpy.array(vec) for vec in data.get('rotate_vectors', [])]
    tvecs = [numpy.array(vec) for vec in data.get('translate_vectors', [])]
    return mtx, dist, rvecs, tvecs


def _undistort(img, mtx, dist, rvecs, tvecs):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    if not any(roi):
        return None

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    if not any(dst.shape):
        return None

    # crop the image
    rx, ry, rw, rh = roi
    dst = cv2.resize(dst[ry:ry + rh, rx:rx + rw].copy(), (w // 2, h // 2))
    return dst


def _mad(arr):
    """
    Median Absolute Deviation of `arr`.

    *Modified to mean absolute deviation*

    Source: https://stackoverflow.com/questions/63991322/median-absolute-deviation-from-numpy-ndarray
    """
    arr = numpy.array(arr)
    median = numpy.median(arr)
    return numpy.mean(numpy.abs(arr - median))


class ImagePairAnalyser:
    """
    For a given move, stores an image of before and of after it.
    Responsible for detecting which squares have changed between those two images.
    """
    def __init__(self, img_prev, img_post, img_no_light):

        self._transfo_matrix = _get_transformation_matrix(img_no_light)
        if self._transfo_matrix is None:
            raise errors.ChessboardNotFoundError("Unable to locate chessboard.")

        self._img_prev = self._normalise_image(img_prev) if img_prev is not None else None
        self._img_post = self._normalise_image(img_post) if img_post is not None else None

        # debug info
        self._working_method = None
        self._changed_squares = []
        self._per_pixel_diff = None
        self._per_square_diff = None

    def save_state(self, directory, name):
        """
        Save the current state of this ImagePairAnalyser to disk.
        Mostly useful for debug purposes.
        """
        path = os.path.join(directory, name)
        prev_path = path + '_prev.jpg'
        post_path = path + '_post.jpg'
        sq_dir = path + '_squares'
        per_square_diff_path = path + '_per_square_diff.jpg'
        per_pixel_diff_path = path + '_per_pixel_diff.jpg'

        if not os.path.exists(sq_dir):
            os.makedirs(sq_dir, exist_ok=True)

        cv2.imwrite(prev_path, self._img_prev)  # type: ignore
        cv2.imwrite(post_path, self._img_post)  # type: ignore

        if self._per_square_diff is not None:
            cv2.imwrite(per_square_diff_path, self._per_square_diff)  # type: ignore
            self._per_square_diff = None
        if self._per_pixel_diff is not None:
            cv2.imwrite(per_pixel_diff_path, self._per_pixel_diff)  # type: ignore
            self._per_pixel_diff = None

        for i, j in numpy.ndindex((8, 8)):
            file = chess.FILE_NAMES[i]
            rank = chess.RANK_NAMES[7 - j]

            fileprefix = os.path.join(sq_dir, f'{file}{rank}_')  # type: ignore
            cv2.imwrite(fileprefix + 'prev.jpg', self._get_square_im(self._img_prev, i, j))  # type: ignore
            cv2.imwrite(fileprefix + 'post.jpg', self._get_square_im(self._img_post, i, j))  # type: ignore

        _json = {
            "working_method": self._working_method,
            "detected_changes": self._changed_squares,
        }
        _json_path = path + 'data.json'
        mode = 'w' if os.path.exists(_json_path) else 'x'
        with open(_json_path, mode=mode) as fs:
            json.dump(_json, fs)  # type: ignore

    @staticmethod
    def _get_square_color(x, y):
        if (x % 2) == (y % 2):
            return ChessColor.WHITE
        return ChessColor.BLACK

    @staticmethod
    def _squares_with_color(color):
        if color == ChessColor.WHITE:
            for i, j in numpy.ndindex((8, 8)):
                if (i % 2) == (j % 2):
                    yield i, j
            return
        for i, j in numpy.ndindex((8, 8)):
            if (i % 2) != (j % 2):
                yield i, j
        return

    def _normalise_image(self, img):
        return _warp_chessboard_image(img, self._transfo_matrix)

    @classmethod
    def _avg_weight(cls, img_square: numpy.ndarray, point: numpy.ndarray):
        size = numpy.array(img_square.shape[:2])
        dist_to_borders = numpy.linalg.norm(1.5 * size + point)  # numpy array element-wise arithmetic

        _pixel_size = max(size)  # assumes that chess squares are squares geometrically as well on the image

        # the following function decreases as its argument approaches 0, which allows
        # us to give less importance to pixels that are closer to the edge of a square.
        if dist_to_borders > (_PIXEL_WEIGHT_RATE * _pixel_size):
            return numpy.sqrt(_PIXEL_WEIGHT_RATE * _pixel_size)
        return numpy.sqrt(dist_to_borders)

    @classmethod
    def _avg_color(cls, img: numpy.ndarray, nchannels=3):
        w, h, *_ = img.shape

        result = numpy.array([0.0 for _ in range(nchannels)])
        denom = 0

        for coords in numpy.ndindex((w, h)):
            weight = cls._avg_weight(img, numpy.array(coords))
            result += weight * img[coords]  # numpy array element-wise arithmetic
            denom += weight

        result /= denom  # (w * h)
        return numpy.abs(result).astype(numpy.uint8)

    @staticmethod
    def _get_square_im(img, x, y):
        _x, _y = _warped_coord(x, y)
        w, h = _WARPED_SQUARE_SIZE

        return img[_y:_y+h, _x:_x+w]

    @classmethod
    def _get_change_mad_threshold(cls, normdiff_per_square, color):
        squares_mean_colors = [normdiff_per_square[i, j] for i, j in cls._squares_with_color(color)]
        mad = _mad(squares_mean_colors)
        median = numpy.median(squares_mean_colors)
        factor = _WHITE_CHANGE_FACTOR if color == ChessColor.WHITE else _BLACK_CHANGE_FACTOR
        return median + (factor * mad), median, mad

    @classmethod
    def _get_change_std_threshold(cls, normdiff_per_square, color):
        squares_mean_colors = [normdiff_per_square[i, j] for i, j in cls._squares_with_color(color)]
        mean = numpy.mean(squares_mean_colors)
        std = numpy.std(squares_mean_colors)
        factor = _WHITE_CHANGE_FACTOR if color == ChessColor.WHITE else _BLACK_CHANGE_FACTOR
        return mean + (factor * std), mean, std

    @classmethod
    def _get_change_mad_mean_threshold(cls, normdiff_per_square, color):
        squares_mean_colors = [normdiff_per_square[i, j] for i, j in cls._squares_with_color(color)]
        mad = _mad(squares_mean_colors)
        mean = numpy.mean(squares_mean_colors)
        factor = _WHITE_CHANGE_FACTOR if color == ChessColor.WHITE else _BLACK_CHANGE_FACTOR
        return mean + (factor * mad), mean, mad

    @classmethod
    def _get_change_std_median_threshold(cls, normdiff_per_square, color):
        squares_mean_colors = [normdiff_per_square[i, j] for i, j in cls._squares_with_color(color)]
        std = numpy.std(squares_mean_colors)
        median = numpy.median(squares_mean_colors)
        factor = _WHITE_CHANGE_FACTOR if color == ChessColor.WHITE else _BLACK_CHANGE_FACTOR
        return median + (factor * std), median, std

    @classmethod
    def _get_change_global_mean_threshold(cls, normdiff_per_square, color):
        mad_thresh, mad_middle, mad_dev = cls._get_change_mad_threshold(normdiff_per_square, color)
        std_thresh, std_middle, std_dev = cls._get_change_std_threshold(normdiff_per_square, color)

        return (
            numpy.mean([mad_thresh, std_thresh]),
            numpy.mean([mad_middle, std_middle]),
            numpy.mean([mad_dev, std_dev]),
        )

    @classmethod
    def _get_change_two_highest_threshold(cls, normdiff_per_square, color):
        mean = numpy.mean(normdiff_per_square)
        median = numpy.median(normdiff_per_square)

        if abs(mean - median) <= (.1 * mean):  # if the median and mean are too close, this means we have no outliers
            thresh = numpy.max(normdiff_per_square) + 1
        else:
            reshaped = numpy.reshape(normdiff_per_square, (normdiff_per_square.size,))
            indices = numpy.argpartition(reshaped, 0)
            second_largest = reshaped[indices[-2]]
            third_largest = reshaped[indices[-3]]
            thresh = numpy.mean([second_largest, third_largest])

        return thresh, mean, numpy.std(normdiff_per_square)

    _move_detect_methods = {}

    @classmethod
    def _get_change_threshold(cls, normdiff_per_square, color, method):
        return cls._move_detect_methods[method](normdiff_per_square, color)

    @classmethod
    def robot_color(cls, img):
        h3_im = cls._get_square_im(img, 7, 5)
        """if not all(cls._avg_color(h3_im) >= 128):
            raise errors.InvalidChessboardError("It seems that the chessboard is positioned the wrong way around.")"""

        h1_im = cls._get_square_im(img, 7, 7)
        if all(cls._avg_color(h1_im) >= 128):  # should use mean, not all()
            return ChessColor.WHITE
        return ChessColor.BLACK

    def push(self, img, start_pos=False):
        self._img_prev = self._img_post
        self._img_post = self._normalise_image(img)
        return self._img_post

    def get_per_square_difference2(self, robot_color):
        """
        Compute norm of the difference of the average color of each square between
        `self._img_prev` and `self._img_post`.

        Problem here is that a change where the piece color is the same as the square color
        is indistinguishable from the random noise.
        """
        if any(x is None for x in (self._img_post, self._img_prev)):  # NOTE: 'None in (self._img_post, self._img_prev)' calls numpy.ndarray.__bool__ which raises ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            return None

        norm_sq_color_avg_diff = numpy.zeros((8, 8), dtype=numpy.float32)

        # for debugging
        self._per_pixel_diff = cv2.absdiff(self._img_prev, self._img_post)
        self._per_square_diff = numpy.zeros((8, 8, 3), dtype=numpy.uint8)
        # end for debugging

        for i, j in numpy.ndindex((8, 8)):

            prev_avg_color = numpy.mean(self._get_square_im(self._img_prev, i, j), axis=(0, 1))
            post_avg_color = numpy.mean(self._get_square_im(self._img_post, i, j), axis=(0, 1))

            diff = prev_avg_color - post_avg_color
            self._per_square_diff[j, i] = diff  # for debugging

            norm_sq_color_avg_diff[j, i] = numpy.linalg.norm(diff)

        return norm_sq_color_avg_diff

    def get_per_square_difference(self, robot_color):
        """
        Compute the per-square difference norm between self._img_prev and self.img_post.
        Also computes normal per-square difference and per-pixel difference for debug
        purposes.
        Very sensitive to slight changes in piece positions.
        """
        if any(x is None for x in (self._img_post, self._img_prev)):  # NOTE: 'None in (self._img_post, self._img_prev)' calls numpy.ndarray.__bool__ which raises ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            return None

        per_pixel_diff = cv2.absdiff(self._img_prev, self._img_post)

        per_square_diff = numpy.zeros((8 * SQUARE_SIZE, 8 * SQUARE_SIZE, 3), dtype=numpy.uint8)
        per_square_norm_diff = numpy.zeros((8, 8), dtype=numpy.float32)

        for i, j in numpy.ndindex((8, 8)):
            if robot_color == ChessColor.BLACK:
                i = 7 - i
                j = 7 - j

            sq_per_pixel_diff = self._get_square_im(per_pixel_diff, i, j)
            mean_diff_color = self._avg_color(sq_per_pixel_diff)
            mean_diff_color_norm = numpy.linalg.norm(mean_diff_color)

            per_square_diff[j*SQUARE_SIZE:(j+1)*SQUARE_SIZE, i*SQUARE_SIZE:(i+1)*SQUARE_SIZE] = mean_diff_color

            per_square_norm_diff[j, i] = mean_diff_color_norm

        self._per_pixel_diff = per_pixel_diff
        self._per_square_diff = per_square_diff

        # cv2.imshow('per pixel diff', per_pixel_diff)
        # cv2.imshow('per square diff', per_square_diff)

        # cv2.waitKey()
        return per_square_norm_diff



    def get_changes(self, per_square_norm_diff, method):
        """
        Attempt to detect the squares that have changed on the board, given the per-square difference using method.
        """
        thresh_white, center_white, dev_white = self._get_change_threshold(per_square_norm_diff, ChessColor.WHITE, method)
        thresh_black, center_black, dev_black = self._get_change_threshold(per_square_norm_diff, ChessColor.BLACK, method)

        result = ChangeList()

        for i, j in numpy.ndindex((8, 8)):
            thresh = (thresh_white if self._get_square_color(i, j) == ChessColor.WHITE
                      else thresh_black)
            if per_square_norm_diff[j, i] > thresh:
                result.set(i, j)

        cv2.waitKey()
        cv2.destroyAllWindows()

        self._changed_squares = [chess.SQUARE_NAMES[_chessmodule_square(change.x, change.y)] for change in result.where_true()]

        return result


ImagePairAnalyser._move_detect_methods['global-mean'] = ImagePairAnalyser._get_change_global_mean_threshold
ImagePairAnalyser._move_detect_methods['std'] = ImagePairAnalyser._get_change_std_threshold
ImagePairAnalyser._move_detect_methods['mad'] = ImagePairAnalyser._get_change_mad_threshold
ImagePairAnalyser._move_detect_methods['std-med'] = ImagePairAnalyser._get_change_std_median_threshold
ImagePairAnalyser._move_detect_methods['mad-mean'] = ImagePairAnalyser._get_change_mad_mean_threshold
ImagePairAnalyser._move_detect_methods['two-highest'] = ImagePairAnalyser._get_change_two_highest_threshold


def _sq_chessmodule(x, y):
    return x, 7 - y


def _chessmodule_square(x, y):
    return chess.square(*_sq_chessmodule(x, y))


class PositionAnalyser:
    def __init__(
            self,
            window,
            cam_id: int | str,
            cam_flags: int | None = None,
            cap_width: int | None = None,
            cap_height: int | None = None,
            cap_delay_ms: int | None = None,
            cap_exposure: int | None = None,
            cap_framecount: int = 1,
            askopenfilename: bool = False,
    ):
        if isinstance(cam_id, str):
            index = _find_cam_id_by_name(cam_id)
            if index >= 0:
                cam_id = index
            else:
                raise errors.CameraNotFoundError(f"Unable to find camera {cam_id!r}.")

        self._cam_id = cam_id
        self._cam_flags = cam_flags
        self._cap_width = cap_width
        self._cap_height = cap_height
        self._cap_delay_ms = cap_delay_ms
        self._cap_exposure = cap_exposure
        self._cap_framecount = cap_framecount
        # take a photo of the empty board with lights turned off.
        _img_no_lights = self.get_image(window, askopenfilename=askopenfilename)
        self._img_analyser = ImagePairAnalyser(None, None, _img_no_lights)
        self._robot_color = None

    def _take_photo(self, prompt_console=False):
        if prompt_console:
            print('Press enter when you are ready.')
            input()

        cam = cv2.VideoCapture()

        if self._cam_flags is None:
            res = cam.open(self._cam_id)
        else:
            res = cam.open(self._cam_id, self._cam_flags)
        if not res:
            return None

        try:
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            if self._cap_width is not None:
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_width)
            if self._cap_height is not None:
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_height)
            if self._cap_exposure is not None:
                cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
                cam.set(cv2.CAP_PROP_EXPOSURE, self._cap_exposure)
            if self._cap_delay_ms is not None:
                # time.sleep(self._cap_delay_ms / 1000)
                start_time = time.time()
                while (time.time() - start_time) <= (self._cap_delay_ms / 1000):
                    cam.read()

            w = math.floor(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = math.floor(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            final_img = Image.empty(w, h, dtype=numpy.uint16) # numpy.zeros((h, w, 3), dtype=numpy.uint16)

            for i in range(self._cap_framecount):
                result, img = cam.read()
                if not result:
                    return None

                if not (any(img.shape) and img.any()):
                    print('frame drop!', file=sys.stderr)

               #  cv2.imshow(f"frame {i}", img)

                final_img += img
            # cv2.waitKey()
        finally:
            cam.release()

        mean_img_float = final_img / self._cap_framecount
        return mean_img_float.astype(numpy.uint8)

    def _guess_move_method(self, board, turn, per_square_diff, method):
        changes = self._img_analyser.get_changes(per_square_diff, method)

        if changes is None:
            return None
        move = changes.guess_move(board, turn)
        if None in move:
            return None

        _from, _to = move

        _sq_from = chess.square(*_sq_chessmodule(*_from))
        _sq_to = chess.square(*_sq_chessmodule(*_to))

        try:
            return board.find_move(_sq_from, _sq_to)
        except chess.IllegalMoveError:
            return None

    @property
    def robot_color(self):
        return self._robot_color

    def start(self, window: Window, askopenfilename=False):
        """
        Start a chess game by taking a photo of the starting position.
        """
        self._cap_exposure = -10
        photo = self.push(window, askopenfilename=askopenfilename, start_pos=True)
        if photo is None:
            return False

        self._robot_color = ImagePairAnalyser.robot_color(photo)
        return True

    def get_image(self, window: Window, askopenfilename=False, start_pos=False):
        """
        Take a photo of the board.
        Returns the photo that was taken.
        If `askopenfilename` is True, asks the user to open an image file instead of
        directly taking a photo.
        """
        print('taking an image...')
        if askopenfilename:
            photo = window.ask_for_user_image()
        else:
            photo = self._take_photo()

        if photo is None:
            return None

        if _IMG_ROTATE_FIX is not None:
            photo = cv2.rotate(photo, _IMG_ROTATE_FIX)  # type: ignore

        return photo

    def push(self, window: Window, askopenfilename=False, start_pos=False):
        """
        Take a photo of the board and store it for move detection.
        Returns the photo that was taken.
        If `askopenfilename` is True, asks the user to open an image file instead of
        directly taking a photo.
        """
        res = self._img_analyser.push(self.get_image(window, askopenfilename=askopenfilename, start_pos=start_pos), start_pos=start_pos)
        return res

    def guess_move(self, board, turn):
        """
        Tries to determine what the most recent move played is.
        """
        self._img_analyser._working_method = None

        try:
            per_square_diff = self._img_analyser.get_per_square_difference(self._robot_color)

            if per_square_diff is None:
                return None
            res = self._guess_move_method(board, turn, per_square_diff, 'global-mean')
            if res is not None:
                self._img_analyser._working_method = 'global-mean'
                return res
            res = self._guess_move_method(board, turn, per_square_diff, 'mad-mean')
            if res is not None:
                self._img_analyser._working_method = 'mad-mean'
                return res
            res = self._guess_move_method(board, turn, per_square_diff, 'std-med')
            if res is not None:
                self._img_analyser._working_method = 'std-mad'
                return res
            res = self._guess_move_method(board, turn, per_square_diff, 'mad')
            if res is not None:
                self._img_analyser._working_method = 'mad'
                return res
            res = self._guess_move_method(board, turn, per_square_diff, 'std')
            if res is not None:
                self._img_analyser._working_method = 'std'
                return res
            res = self._guess_move_method(board, turn, per_square_diff, 'two-highest')
            if res is not None:
                self._img_analyser._working_method = 'two-highest'
            return res


        finally:
            debug_info.save_move(self._img_analyser)

