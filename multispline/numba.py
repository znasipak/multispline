"""Numba-compatible evaluation of multispline splines.

The spline *fitting* (computing the polynomial coefficients) is performed once in
the C++/Cython backend and cannot be called from within a ``numba`` ``nopython``
function.  Spline *evaluation*, however, is just an interval lookup followed by a
nested Horner polynomial over a plain ``float64`` coefficient array -- exactly the
kind of arithmetic ``numba`` compiles efficiently.

This module exposes that evaluation in two layers:

1. Stand-alone ``@njit`` kernels (``cubic_eval_d``, ``bicubic_eval_d``,
   ``tricubic_eval_d``, ``quadcubic_eval_d`` and friends) that take the coefficient
   array plus the grid parameters explicitly.  They can be called from inside your
   own ``@njit`` functions.

2. ``jitclass`` wrappers (``CubicSplineNumba`` ...) that bundle the coefficients and
   grid parameters together so you can write ``spline.eval(x)`` inside ``@njit`` code.
   Build one from an existing spline with ``spline.to_numba()``.

Example
-------
>>> import numpy as np
>>> from numba import njit
>>> from multispline.spline import TricubicSpline
>>> spl = TricubicSpline(x, y, z, f)
>>> nspl = spl.to_numba()              # build once, outside njit
>>> @njit
... def integrate(nspl, pts):
...     total = 0.0
...     for n in range(pts.shape[0]):
...         total += nspl.eval(pts[n, 0], pts[n, 1], pts[n, 2])
...     return total

The coefficients follow the same *normalized* local-coordinate convention used by
the C++ backend, so the kernels reproduce ``CubicSpline.eval``/``deriv*`` etc. to
machine precision.
"""

try:
    import numpy as np
    from numba import njit, int64, float64
    from numba.experimental import jitclass
except ImportError as exc:  # pragma: no cover - exercised only without numba
    raise ImportError(
        "multispline.numba requires the optional 'numba' dependency. "
        "Install it with `pip install multispline[numba]` or `pip install numba`."
    ) from exc


# ---------------------------------------------------------------------------
# low-level device functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def _locate(x, x0, dx, n):
    """Return the interval index ``i`` (clamped to ``[0, n-1]``) and the
    normalized local coordinate ``t = (x - x0)/dx - i``.

    Mirrors ``findInterval`` in the C++ backend, including the clamping that
    extrapolates with the nearest interval for out-of-domain points.
    """
    i = int((x - x0) / dx)
    if i < 0:
        i = 0
    elif i > n - 1:
        i = n - 1
    t = (x - x0) / dx - i
    return i, t


@njit(cache=True)
def _seval(c0, c1, c2, c3, t, d):
    """Evaluate the cubic ``c0 + c1 t + c2 t^2 + c3 t^3`` (or its ``d``-th
    derivative with respect to the *normalized* coordinate ``t``) via Horner's
    method.  ``d`` is 0, 1 or 2.
    """
    if d == 0:
        return c0 + t * (c1 + t * (c2 + t * c3))
    elif d == 1:
        return c1 + t * (2.0 * c2 + 3.0 * c3 * t)
    else:
        return 2.0 * c2 + 6.0 * c3 * t


# ---------------------------------------------------------------------------
# cubic (1D)
# ---------------------------------------------------------------------------

@njit(cache=True)
def cubic_eval_d(c, x0, dx, nx, x, dox):
    """Evaluate a 1D cubic spline (or its ``dox``-th derivative) at ``x``.

    ``c`` has shape ``(nx, 4)``.
    """
    i, t = _locate(x, x0, dx, nx)
    val = _seval(c[i, 0], c[i, 1], c[i, 2], c[i, 3], t, dox)
    return val / dx ** dox


@njit(cache=True)
def cubic_eval(c, x0, dx, nx, x):
    return cubic_eval_d(c, x0, dx, nx, x, 0)


@njit(cache=True)
def cubic_deriv(c, x0, dx, nx, x):
    return cubic_eval_d(c, x0, dx, nx, x, 1)


@njit(cache=True)
def cubic_deriv2(c, x0, dx, nx, x):
    return cubic_eval_d(c, x0, dx, nx, x, 2)


# ---------------------------------------------------------------------------
# bicubic (2D)
# ---------------------------------------------------------------------------

@njit(cache=True)
def bicubic_eval_d(c, x0, dx, nx, y0, dy, ny, x, y, dox, doy):
    """Evaluate a 2D bicubic spline (or a mixed partial derivative of order
    ``dox`` in x and ``doy`` in y) at ``(x, y)``.

    ``c`` has shape ``(nx, ny, 4, 4)`` indexed ``[i, j, mx, my]``.
    """
    i, tx = _locate(x, x0, dx, nx)
    j, ty = _locate(y, y0, dy, ny)
    z0 = _seval(c[i, j, 0, 0], c[i, j, 0, 1], c[i, j, 0, 2], c[i, j, 0, 3], ty, doy)
    z1 = _seval(c[i, j, 1, 0], c[i, j, 1, 1], c[i, j, 1, 2], c[i, j, 1, 3], ty, doy)
    z2 = _seval(c[i, j, 2, 0], c[i, j, 2, 1], c[i, j, 2, 2], c[i, j, 2, 3], ty, doy)
    z3 = _seval(c[i, j, 3, 0], c[i, j, 3, 1], c[i, j, 3, 2], c[i, j, 3, 3], ty, doy)
    val = _seval(z0, z1, z2, z3, tx, dox)
    return val / dx ** dox / dy ** doy


# ---------------------------------------------------------------------------
# tricubic (3D)
# ---------------------------------------------------------------------------

@njit(cache=True)
def tricubic_eval_d(c, x0, dx, nx, y0, dy, ny, z0, dz, nz, x, y, z, dox, doy, doz):
    """Evaluate a 3D tricubic spline (or a mixed partial derivative of orders
    ``dox, doy, doz``) at ``(x, y, z)``.

    ``c`` has shape ``(nx, ny, nz, 4, 4, 4)`` indexed ``[i, j, k, mx, my, mz]``.
    """
    i, tx = _locate(x, x0, dx, nx)
    j, ty = _locate(y, y0, dy, ny)
    k, tz = _locate(z, z0, dz, nz)
    yv = np.empty(4)
    for mx in range(4):
        zv0 = _seval(c[i, j, k, mx, 0, 0], c[i, j, k, mx, 0, 1], c[i, j, k, mx, 0, 2], c[i, j, k, mx, 0, 3], tz, doz)
        zv1 = _seval(c[i, j, k, mx, 1, 0], c[i, j, k, mx, 1, 1], c[i, j, k, mx, 1, 2], c[i, j, k, mx, 1, 3], tz, doz)
        zv2 = _seval(c[i, j, k, mx, 2, 0], c[i, j, k, mx, 2, 1], c[i, j, k, mx, 2, 2], c[i, j, k, mx, 2, 3], tz, doz)
        zv3 = _seval(c[i, j, k, mx, 3, 0], c[i, j, k, mx, 3, 1], c[i, j, k, mx, 3, 2], c[i, j, k, mx, 3, 3], tz, doz)
        yv[mx] = _seval(zv0, zv1, zv2, zv3, ty, doy)
    val = _seval(yv[0], yv[1], yv[2], yv[3], tx, dox)
    return val / dx ** dox / dy ** doy / dz ** doz


# ---------------------------------------------------------------------------
# quadcubic (4D)
# ---------------------------------------------------------------------------

@njit(cache=True)
def quadcubic_eval_d(c, w0, dw, nw, x0, dx, nx, y0, dy, ny, z0, dz, nz,
                     w, x, y, z, dow, dox, doy, doz):
    """Evaluate a 4D quadcubic spline (or a mixed partial derivative of orders
    ``dow, dox, doy, doz``) at ``(w, x, y, z)``.

    ``c`` has shape ``(nw, nx, ny, nz, 4, 4, 4, 4)`` indexed
    ``[h, i, j, k, mw, mx, my, mz]``.
    """
    h, tw = _locate(w, w0, dw, nw)
    i, tx = _locate(x, x0, dx, nx)
    j, ty = _locate(y, y0, dy, ny)
    k, tz = _locate(z, z0, dz, nz)
    xv = np.empty(4)
    for mw in range(4):
        yv = np.empty(4)
        for mx in range(4):
            zv0 = _seval(c[h, i, j, k, mw, mx, 0, 0], c[h, i, j, k, mw, mx, 0, 1], c[h, i, j, k, mw, mx, 0, 2], c[h, i, j, k, mw, mx, 0, 3], tz, doz)
            zv1 = _seval(c[h, i, j, k, mw, mx, 1, 0], c[h, i, j, k, mw, mx, 1, 1], c[h, i, j, k, mw, mx, 1, 2], c[h, i, j, k, mw, mx, 1, 3], tz, doz)
            zv2 = _seval(c[h, i, j, k, mw, mx, 2, 0], c[h, i, j, k, mw, mx, 2, 1], c[h, i, j, k, mw, mx, 2, 2], c[h, i, j, k, mw, mx, 2, 3], tz, doz)
            zv3 = _seval(c[h, i, j, k, mw, mx, 3, 0], c[h, i, j, k, mw, mx, 3, 1], c[h, i, j, k, mw, mx, 3, 2], c[h, i, j, k, mw, mx, 3, 3], tz, doz)
            yv[mx] = _seval(zv0, zv1, zv2, zv3, ty, doy)
        xv[mw] = _seval(yv[0], yv[1], yv[2], yv[3], tx, dox)
    val = _seval(xv[0], xv[1], xv[2], xv[3], tw, dow)
    return val / dw ** dow / dx ** dox / dy ** doy / dz ** doz


# ---------------------------------------------------------------------------
# jitclass wrappers -- usable inside @njit as `spline.eval(x)`
# ---------------------------------------------------------------------------

_cubic_spec = [
    ("c", float64[:, :]),
    ("x0", float64), ("dx", float64), ("nx", int64),
]


@jitclass(_cubic_spec)
class CubicSplineNumba:
    def __init__(self, c, x0, dx, nx):
        self.c = c
        self.x0 = x0
        self.dx = dx
        self.nx = nx

    def eval(self, x):
        return cubic_eval_d(self.c, self.x0, self.dx, self.nx, x, 0)

    def deriv(self, x):
        return cubic_eval_d(self.c, self.x0, self.dx, self.nx, x, 1)

    def deriv2(self, x):
        return cubic_eval_d(self.c, self.x0, self.dx, self.nx, x, 2)


_bicubic_spec = [
    ("c", float64[:, :, :, :]),
    ("x0", float64), ("dx", float64), ("nx", int64),
    ("y0", float64), ("dy", float64), ("ny", int64),
]


@jitclass(_bicubic_spec)
class BicubicSplineNumba:
    def __init__(self, c, x0, dx, nx, y0, dy, ny):
        self.c = c
        self.x0 = x0
        self.dx = dx
        self.nx = nx
        self.y0 = y0
        self.dy = dy
        self.ny = ny

    def eval(self, x, y):
        return bicubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, x, y, 0, 0)

    def deriv_x(self, x, y):
        return bicubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, x, y, 1, 0)

    def deriv_y(self, x, y):
        return bicubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, x, y, 0, 1)

    def deriv_xx(self, x, y):
        return bicubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, x, y, 2, 0)

    def deriv_yy(self, x, y):
        return bicubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, x, y, 0, 2)

    def deriv_xy(self, x, y):
        return bicubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, x, y, 1, 1)


_tricubic_spec = [
    ("c", float64[:, :, :, :, :, :]),
    ("x0", float64), ("dx", float64), ("nx", int64),
    ("y0", float64), ("dy", float64), ("ny", int64),
    ("z0", float64), ("dz", float64), ("nz", int64),
]


@jitclass(_tricubic_spec)
class TricubicSplineNumba:
    def __init__(self, c, x0, dx, nx, y0, dy, ny, z0, dz, nz):
        self.c = c
        self.x0 = x0
        self.dx = dx
        self.nx = nx
        self.y0 = y0
        self.dy = dy
        self.ny = ny
        self.z0 = z0
        self.dz = dz
        self.nz = nz

    def eval(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 0, 0, 0)

    def deriv_x(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 1, 0, 0)

    def deriv_y(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 0, 1, 0)

    def deriv_z(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 0, 0, 1)

    def deriv_xx(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 2, 0, 0)

    def deriv_yy(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 0, 2, 0)

    def deriv_zz(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 0, 0, 2)

    def deriv_xy(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 1, 1, 0)

    def deriv_xz(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 1, 0, 1)

    def deriv_yz(self, x, y, z):
        return tricubic_eval_d(self.c, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, x, y, z, 0, 1, 1)


_quadcubic_spec = [
    ("c", float64[:, :, :, :, :, :, :, :]),
    ("w0", float64), ("dw", float64), ("nw", int64),
    ("x0", float64), ("dx", float64), ("nx", int64),
    ("y0", float64), ("dy", float64), ("ny", int64),
    ("z0", float64), ("dz", float64), ("nz", int64),
]


@jitclass(_quadcubic_spec)
class QuadcubicSplineNumba:
    def __init__(self, c, w0, dw, nw, x0, dx, nx, y0, dy, ny, z0, dz, nz):
        self.c = c
        self.w0 = w0
        self.dw = dw
        self.nw = nw
        self.x0 = x0
        self.dx = dx
        self.nx = nx
        self.y0 = y0
        self.dy = dy
        self.ny = ny
        self.z0 = z0
        self.dz = dz
        self.nz = nz

    def eval(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 0, 0, 0)

    def deriv_w(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 1, 0, 0, 0)

    def deriv_x(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 1, 0, 0)

    def deriv_y(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 0, 1, 0)

    def deriv_z(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 0, 0, 1)

    def deriv_ww(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 2, 0, 0, 0)

    def deriv_xx(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 2, 0, 0)

    def deriv_yy(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 0, 2, 0)

    def deriv_zz(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 0, 0, 2)

    def deriv_wx(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 1, 1, 0, 0)

    def deriv_wy(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 1, 0, 1, 0)

    def deriv_wz(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 1, 0, 0, 1)

    def deriv_xy(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 1, 1, 0)

    def deriv_xz(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 1, 0, 1)

    def deriv_yz(self, w, x, y, z):
        return quadcubic_eval_d(self.c, self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, w, x, y, z, 0, 0, 1, 1)
