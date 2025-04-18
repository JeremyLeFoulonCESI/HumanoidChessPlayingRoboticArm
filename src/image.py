"""
Wrapper around numpy arrays that provides an interface to certain openCV functions.
Not fully taken advantage of yet, but has a high potential of simplifying a lot of code.
"""


import cv2
import numpy
import threading

from enum import IntEnum


class ColorFormat(IntEnum):
    BGR = 0
    RGB = 1
    GRAY = 2
    HSV = 3
    HLS = 4
    LUV = 5


_channelcount = {
    fmt: 3 for fmt in ColorFormat._member_map_.values()
}
_channelcount[ColorFormat.GRAY] = 1


def _fmt_convert_to_cv2(_from: ColorFormat, _to: ColorFormat):
    if _from == _to:
        return None

    _from_name = _from.name
    _to_name = _to.name

    cvt_name = f'COLOR_{_from_name}2{_to_name}'
    return getattr(cv2, cvt_name, None)


class Image(numpy.ndarray):
    """
    Array type that represents an image.
    Provides an interface to some OpenCV functions.
    """

    def __new__(cls, src=None, /, w=None, h=None, fmt=ColorFormat.BGR, dtype=None, offset=0, strides=None, order=None):
        shape = (h, w, _channelcount[fmt])
        if isinstance(src, numpy.ndarray):
            shape = src.shape
            dtype = src.dtype
            strides = src.strides
            offset = 0
        self = super().__new__(cls, shape, dtype=dtype, buffer=src, offset=offset, strides=strides, order=order)
        self._fmt = fmt
        return self

    def __array_finalize__(self, obj, /):
        if isinstance(obj, Image):
            self._fmt = obj._fmt
        elif not hasattr(self, '_fmt'):
            self._fmt = ColorFormat.BGR

    def im_show(self, title):
        """
        Direct interface to cv2.imshow(title, image)
        """
        cv2.imshow(title, self)

    def show(self, title):
        """
        Shows the image and returns the ShowGroup that manages its window.
        This function is not blocking.

        Using `image.show(title).wait()` will show the image and wait for the
        user to close it.

        Note that using this automatically makes the program and current
        process multithreaded.
        """
        return show_group(**{title: self})

    def cvt_color(self, new_fmt: ColorFormat):
        """
        Convert the image from one color format to another.
        """
        return type(self)(cv2.cvtColor(self, _fmt_convert_to_cv2(self._fmt, new_fmt)))

    def resize(self, new_w, new_h):
        """
        Change the size of the image.
        """
        new_shape = (new_h, new_w, *self.shape[2:])
        return type(self)(cv2.resize(self, new_shape))

    def scale_to(self, /, *, w: float | None = None, h: float | None = None):
        """
        Rescale the image to either a specified width or a specified height, but not both at the same time.
        """
        if (w is None) and (h is None):
            return None
        if w is None:
            factor = h / self.height
        else:
            factor = w / self.width

        new_w = self.width * factor
        new_h = self.height * factor

        return self.resize(new_w, new_h)

    def write(self, filename):
        """
        Write the image to the specified file.
        """
        return cv2.imwrite(filename, self)

    @classmethod
    def _from_nparray(cls, arr, *args, fmt, **kwargs):
        return cls(arr, *args, fmt=fmt, **kwargs)

    @classmethod
    def read(cls, filename, fmt=ColorFormat.BGR, **kwargs):
        """
        Read and return an image from the specified file.
        The image format defaults to BGR.
        """
        args = [filename]
        if fmt != ColorFormat.BGR:
            args.append(_fmt_convert_to_cv2(ColorFormat.BGR, fmt))
        return cls._from_nparray(cv2.imread(*args, **kwargs), fmt=fmt)

    @classmethod
    def full(cls, width, height, value, fmt=ColorFormat.BGR, **kwargs):
        """
        Return an image where all channels of all pixels have the specified value.
        """
        shape = [height, width]
        if _channelcount[fmt] > 1:
            shape.append(_channelcount[fmt])
        return cls._from_nparray(numpy.full(shape, value, **kwargs), fmt=fmt)

    @classmethod
    def empty(cls, width, height, fmt=ColorFormat.BGR, **kwargs):
        """
        Return an image where all channels of all pixels have the value 0.
        """
        return cls(w=width, h=height, fmt=fmt, **kwargs)

    @property
    def color_channels(self):
        """
        Number of color channels for the current color format.
        """
        if len(self.shape) <= 2:
            return 1
        return self.shape[2]

    @property
    def format(self):
        """
        Color format of the image.
        """
        return self._fmt

    @property
    def width(self):
        """
        Width of the image in pixels.
        """
        if len(self.shape) <= 1:
            return 1
        return self.shape[1]

    @property
    def height(self):
        """
        Height of the image in pixels.
        """
        if not len(self.shape):
            return 1
        return self.shape[0]


class _ShownGroup:
    """
    Represents a set of images that are currently being displayed on the screen.
    """
    def __init__(self, **images: Image):
        self._images = images
        self._thread = threading.Thread(target=self._displaythread)
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if None in (exc_type, exc_val):
            self.wait(close=True)
            return False

        self.close_windows()
        return False

    def wait(self, /, *, close=False, **kwargs):
        """
        Wait for all the currently visible windows to be closed.
        """
        self._thread.join(**kwargs)
        if not self._thread.is_alive() and close:
            self.close_windows()
            return True
        return not self._thread.is_alive()

    def close_windows(self):
        """
        Close all the windows associated with this group.
        """
        for name in self._images.keys():
            try:
                cv2.destroyWindow(name)
            except:
                pass

    def _displaythread(self):
        for title, img in self._images.items():
            img.im_show(title)
        while cv2.waitKey() >= 0:
            pass


def show_group(**images):
    """
    Returns a ShowGroup object that handles displaying the images.

    Using this in a with statement will wait for all windows to be
    closed by the user before leaving the with statement. If an error
    occurs in the with statement, all windows associated with the current
    group are closed immediately and the exception is propagated.
    """
    return _ShownGroup(**images)