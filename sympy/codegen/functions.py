from sympy.core.function import ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log


def _powm1(b, e):
    return Pow(b, e) - S.One


class powm1(Function):
    """Represents Pow(base, exponent) - S.One, useful for avoiding cancellation.

    """
    nargs = 2

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        b, e = self.args
        if argindex == 1:
            return Pow(b, e-1)*e
        elif argindex == 2:
            return Pow(b, e)*log(b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Pow(self, b, e, **kwargs):
        return _powm1(b, e)

    _eval_expand_func = _eval_rewrite_as_Pow

    def _eval_is_real(self):
        return Pow(*self.args).is_real

    def _eval_is_finite(self):
        return Pow(*self.args).is_finite

    def _eval_is_zero(self):
        return self.args[0] == S.One or (self.args[1].is_zero and self.args[0].is_nonzero)


def _sqrt1pm1(x):
    return sqrt(S.One + x) - S.One

class sqrt1pm1(Function):
    nargs = 1

    def fdiff(self, argindex=1):
        if argindex == 1:
            return Pow(2*sqrt(S.One+self.args[0]), -1)
        else:
            raise ArgumentIndexError(self, argindex)
