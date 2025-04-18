from numbers import Rational
from fractions import Fraction


class MutableFraction(Rational):
    @property
    def numerator(self):
        return self._numerator

    @numerator.setter
    def numerator(self, value):
        self._numerator = value

    @property
    def denominator(self):
        return self._denominator

    @denominator.setter
    def denominator(self, value):
        self._denominator = value

    def __trunc__(self):
        return self._fraction_op('trunc')

    def __floor__(self):
        return self._fraction_op('floor')

    def __ceil__(self):
        return self._fraction_op('ceil')

    def __round__(self, ndigits=None):
        return self._fraction_op('round', ndigits=ndigits)

    def __floordiv__(self, other):
        return self._fraction_op('floordiv', other)

    def __ifloordiv__(self, other):
        return self._inplace_fraction_op('floordiv', other)

    def __rfloordiv__(self, other):
        return self._fraction_op('rfloordiv', other)

    def __mod__(self, other):
        return self._fraction_op('mod', other)

    def __imod__(self, other):
        return self._inplace_fraction_op('mod', other)

    def __rmod__(self, other):
        return self._fraction_op('rmod', other)

    def __lt__(self, other):
        return self._fraction_op('lt', other)

    def __le__(self, other):
        return self._fraction_op('le', other)

    def __pos__(self):
        return self._fraction_op('pos')

    def __neg__(self):
        return self._fraction_op('neg')

    def __add__(self, other):
        return self._fraction_op('add', other)

    def __iadd__(self, other):
        return self._inplace_fraction_op('add', other)

    def __radd__(self, other):
        return self._fraction_op('radd', other)

    def __mul__(self, other):
        return self._fraction_op('mul', other)

    def __imul__(self, other):
        return self._inplace_fraction_op('mul', other)

    def __rmul__(self, other):
        return self._fraction_op('rmul', other)

    def __truediv__(self, other):
        return self._fraction_op('truediv', other)

    def __itruediv__(self, other):
        return self._inplace_fraction_op('truediv', other)

    def __rtruediv__(self, other):
        return self._fraction_op('rtruediv', other)

    def __pow__(self, exponent):
        return self._fraction_op('pow', exponent)

    def __ipow__(self, other):
        return self._fraction_op('pow', other)

    def __rpow__(self, base):
        return self._fraction_op('rpow', base)

    def __abs__(self):
        return self._fraction_op('abs')

    def __eq__(self, other):
        fother = Fraction(other)
        return self._numerator == fother.numerator and self._denominator == fother.denominator

    def __hash__(self):
        return Fraction(self._numerator, self._denominator).__hash__()

    def __init__(self, numerator=0, denominator=1, /):
        self._numerator = numerator
        self._denominator = denominator

    def __repr__(self):
        return f'{self.__class__.__name__}({self._numerator}, {self._denominator})'

    def __str__(self):
        return

    def _raw_fraction_op(self, op, *args, **kwargs):
        op = getattr(Fraction, f'__{op}__')
        return op(Fraction(self._numerator, self._denominator), *args, **kwargs)

    def _fraction_op(self, op, *args, **kwargs):
        res = self._raw_fraction_op(op, *args, **kwargs)
        if isinstance(res, Fraction):
            return type(self)(res.numerator, res.denominator)
        return res

    def _inplace_fraction_op(self, op, *args, **kwargs):
        res = self._raw_fraction_op(op, *args, **kwargs)
        if not isinstance(res, Rational):
            res = Fraction(res)
        self._numerator = res.numerator
        self._denominator = res.denominator
        return self

