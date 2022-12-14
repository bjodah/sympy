from itertools import product
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos
from sympy.core.numbers import pi
from sympy.codegen.scipy_nodes import cosm1, powm1

x, y, z = symbols('x y z')


def test_cosm1():
    cm1_xy = cosm1(x*y)
    ref_xy = cos(x*y) - 1
    for wrt, deriv_order in product([x, y, z], range(3)):
        assert (
            cm1_xy.diff(wrt, deriv_order) -
            ref_xy.diff(wrt, deriv_order)
        ).rewrite(cos).simplify() == 0

    expr_minus2 = cosm1(pi)
    assert expr_minus2.rewrite(cos) == -2
    assert cosm1(3.14).simplify() == cosm1(3.14)  # cannot simplify with 3.14
    assert cosm1(pi/2).simplify() == -1
    assert (1/cos(x) - 1 + cosm1(x)/cos(x)).simplify() == 0
    assert str(cosm1(1)) == "cosm1(1)"

    # Series expansion
    for expr in [x, cos(x**2+1)]:
        for x0 in [0, 1, 2]:
            a = cosm1(expr).series(x, x0)
            b = (cos(expr) - 1).series(x, x0)
            if x0 == 0 and expr == x:
                assert a == b
            else:
                assert (a - b).expand(func=True).removeO() == 0


def test_powm1():
    cases = {
            powm1(x, y): x**y - 1,
            powm1(x*y, z): (x*y)**z - 1,
            powm1(x, y*z): x**(y*z)-1,
            powm1(x*y*z, x*y*z): (x*y*z)**(x*y*z)-1
    }
    for pm1_e, ref_e in cases.items():
        for wrt, deriv_order in product([x, y, z], range(3)):
            der = pm1_e.diff(wrt, deriv_order)
            ref = ref_e.diff(wrt, deriv_order)
            delta = (der - ref).rewrite(Pow)
            assert delta.simplify() == 0

    eulers_constant_m1 = powm1(x, 1/log(x))
    assert eulers_constant_m1.rewrite(Pow) == exp(1) - 1
    assert eulers_constant_m1.simplify() == exp(1) - 1

    # Series expansion
    for expr_x in [x, cos(x**2+1)]:
        for around in [0, 1, 2]:
            a = powm1(expr_x, y).series(x, around).expand(func=True)
            b = (expr_x**y - 1).series(x, around)
            assert a == b

            expr_y = expr_x.subs({x: y})
            a = powm1(x, expr_y).series(y, around).expand(func=True)
            b = (x**expr_y - 1).series(y, around)
            assert a == b
