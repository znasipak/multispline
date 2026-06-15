"""JAX-compatible evaluation of multispline splines.

The spline *fitting* (computing the polynomial coefficients) is performed once in
the C++/Cython backend and is not traced by JAX.  Spline *evaluation*, however, is
just an interval lookup followed by a polynomial contraction over a plain
coefficient array -- pure arithmetic that JAX can ``jit``, ``vmap`` and
differentiate.

Build a spline as usual, then call ``spline.to_jax()`` to obtain a lightweight,
registered-pytree view (a ``NamedTuple``) whose ``eval``/``deriv_*`` methods can be
used inside ``jax.jit``/``jax.vmap``/``jax.grad`` and composed into larger
differentiable computations.

Derivatives are obtained by **automatic differentiation** of the value kernel.
Because the spline is piecewise-polynomial, AD reproduces the exact analytic
derivatives (to machine precision) -- the same values the C++ ``deriv_*`` methods
return.

Precision
---------
JAX defaults to 32-bit floats.  Spline coefficients are double precision, so for
results that match the C++ backend you must enable 64-bit mode *before* any JAX
computation runs::

    import jax
    jax.config.update("jax_enable_x64", True)

``to_jax`` emits a warning if x64 is disabled.

Example
-------
>>> import jax
>>> jax.config.update("jax_enable_x64", True)
>>> import jax.numpy as jnp
>>> from multispline.spline import TricubicSpline
>>> spl = TricubicSpline(x, y, z, f)
>>> jspl = spl.to_jax()                       # build once
>>> jspl.eval(0.3, 0.4, 0.5)                  # scalar
>>> batched = jax.vmap(jspl.eval)             # vectorize over points
>>> batched(xs, ys, zs)
>>> jax.grad(jspl.eval, argnums=0)(0.3, 0.4, 0.5)   # == jspl.deriv_x(...)
"""
import typing
import warnings

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised only without jax
    raise ImportError(
        "multispline.jax requires the optional 'jax' dependency. "
        "Install it with `pip install multispline[jax]` or `pip install jax`."
    ) from exc


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _basis(t):
    """Power basis ``[1, t, t**2, t**3]`` for a normalized local coordinate."""
    return jnp.stack([jnp.ones_like(t), t, t * t, t * t * t])


def _locate(x, x0, dx, n):
    """Interval index ``i`` (clamped to ``[0, n-1]``) and normalized local
    coordinate ``t = (x - x0)/dx - i``.

    ``floor``/``clip`` are piecewise constant, so under AD the only gradient
    path is through ``t`` with ``dt/dx = 1/dx`` -- which yields exactly the
    analytic spline derivatives.
    """
    val = (x - x0) / dx
    iv = jnp.clip(jnp.floor(val), 0.0, n - 1.0)
    i = iv.astype(jnp.int32)
    t = val - iv
    return i, t


def _d(f, *argnums):
    """Compose ``jax.grad`` over the given argnums and jit the result."""
    g = f
    for a in argnums:
        g = jax.grad(g, argnums=a)
    return jax.jit(g)


def _x64_enabled():
    try:
        return bool(jax.config.jax_enable_x64)
    except Exception:  # pragma: no cover - defensive across jax versions
        try:
            return bool(jax.config.read("jax_enable_x64"))
        except Exception:
            return True


def _warn_if_no_x64():
    if not _x64_enabled():
        warnings.warn(
            "JAX 64-bit mode is disabled, so multispline.jax evaluation runs in "
            "float32 and will not match the C++ backend to double precision. "
            "Enable it with `jax.config.update('jax_enable_x64', True)` before any "
            "JAX computation.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# value kernels (one per dimension) -- everything else is AD of these
# ---------------------------------------------------------------------------

def _cubic_value(spline, x):
    c = spline.coeffs                                   # (nx, 4)
    i, t = _locate(x, spline.x0, spline.dx, c.shape[0])
    return jnp.dot(c[i], _basis(t))


def _bicubic_value(spline, x, y):
    c = spline.coeffs                                   # (nx, ny, 4, 4)
    i, tx = _locate(x, spline.x0, spline.dx, c.shape[0])
    j, ty = _locate(y, spline.y0, spline.dy, c.shape[1])
    return jnp.einsum("ab,a,b->", c[i, j], _basis(tx), _basis(ty))


def _tricubic_value(spline, x, y, z):
    c = spline.coeffs                                   # (nx, ny, nz, 4, 4, 4)
    i, tx = _locate(x, spline.x0, spline.dx, c.shape[0])
    j, ty = _locate(y, spline.y0, spline.dy, c.shape[1])
    k, tz = _locate(z, spline.z0, spline.dz, c.shape[2])
    return jnp.einsum("abc,a,b,c->", c[i, j, k], _basis(tx), _basis(ty), _basis(tz))


def _quadcubic_value(spline, w, x, y, z):
    c = spline.coeffs                                   # (nw, nx, ny, nz, 4, 4, 4, 4)
    h, tw = _locate(w, spline.w0, spline.dw, c.shape[0])
    i, tx = _locate(x, spline.x0, spline.dx, c.shape[1])
    j, ty = _locate(y, spline.y0, spline.dy, c.shape[2])
    k, tz = _locate(z, spline.z0, spline.dz, c.shape[3])
    return jnp.einsum("abcd,a,b,c,d->", c[h, i, j, k],
                      _basis(tw), _basis(tx), _basis(ty), _basis(tz))


# ---------------------------------------------------------------------------
# jitted value + derivative functions (AD)
# argnum convention: spline is 0; coordinates follow.
# ---------------------------------------------------------------------------

# cubic: x -> 1
_cubic_eval = jax.jit(_cubic_value)
_cubic_deriv = _d(_cubic_value, 1)
_cubic_deriv2 = _d(_cubic_value, 1, 1)

# bicubic: x -> 1, y -> 2
_bicubic_eval = jax.jit(_bicubic_value)
_bicubic_dx = _d(_bicubic_value, 1)
_bicubic_dy = _d(_bicubic_value, 2)
_bicubic_dxx = _d(_bicubic_value, 1, 1)
_bicubic_dyy = _d(_bicubic_value, 2, 2)
_bicubic_dxy = _d(_bicubic_value, 1, 2)

# tricubic: x -> 1, y -> 2, z -> 3
_tricubic_eval = jax.jit(_tricubic_value)
_tricubic_dx = _d(_tricubic_value, 1)
_tricubic_dy = _d(_tricubic_value, 2)
_tricubic_dz = _d(_tricubic_value, 3)
_tricubic_dxx = _d(_tricubic_value, 1, 1)
_tricubic_dyy = _d(_tricubic_value, 2, 2)
_tricubic_dzz = _d(_tricubic_value, 3, 3)
_tricubic_dxy = _d(_tricubic_value, 1, 2)
_tricubic_dxz = _d(_tricubic_value, 1, 3)
_tricubic_dyz = _d(_tricubic_value, 2, 3)

# quadcubic: w -> 1, x -> 2, y -> 3, z -> 4
_quadcubic_eval = jax.jit(_quadcubic_value)
_quadcubic_dw = _d(_quadcubic_value, 1)
_quadcubic_dx = _d(_quadcubic_value, 2)
_quadcubic_dy = _d(_quadcubic_value, 3)
_quadcubic_dz = _d(_quadcubic_value, 4)
_quadcubic_dww = _d(_quadcubic_value, 1, 1)
_quadcubic_dxx = _d(_quadcubic_value, 2, 2)
_quadcubic_dyy = _d(_quadcubic_value, 3, 3)
_quadcubic_dzz = _d(_quadcubic_value, 4, 4)
_quadcubic_dwx = _d(_quadcubic_value, 1, 2)
_quadcubic_dwy = _d(_quadcubic_value, 1, 3)
_quadcubic_dwz = _d(_quadcubic_value, 1, 4)
_quadcubic_dxy = _d(_quadcubic_value, 2, 3)
_quadcubic_dxz = _d(_quadcubic_value, 2, 4)
_quadcubic_dyz = _d(_quadcubic_value, 3, 4)


# ---------------------------------------------------------------------------
# registered-pytree wrappers (NamedTuples are pytrees automatically)
# ---------------------------------------------------------------------------

class CubicSplineJax(typing.NamedTuple):
    coeffs: typing.Any
    x0: typing.Any
    dx: typing.Any

    def eval(self, x):
        return _cubic_eval(self, x)

    def deriv(self, x):
        return _cubic_deriv(self, x)

    def deriv2(self, x):
        return _cubic_deriv2(self, x)


class BicubicSplineJax(typing.NamedTuple):
    coeffs: typing.Any
    x0: typing.Any
    dx: typing.Any
    y0: typing.Any
    dy: typing.Any

    def eval(self, x, y):
        return _bicubic_eval(self, x, y)

    def deriv_x(self, x, y):
        return _bicubic_dx(self, x, y)

    def deriv_y(self, x, y):
        return _bicubic_dy(self, x, y)

    def deriv_xx(self, x, y):
        return _bicubic_dxx(self, x, y)

    def deriv_yy(self, x, y):
        return _bicubic_dyy(self, x, y)

    def deriv_xy(self, x, y):
        return _bicubic_dxy(self, x, y)


class TricubicSplineJax(typing.NamedTuple):
    coeffs: typing.Any
    x0: typing.Any
    dx: typing.Any
    y0: typing.Any
    dy: typing.Any
    z0: typing.Any
    dz: typing.Any

    def eval(self, x, y, z):
        return _tricubic_eval(self, x, y, z)

    def deriv_x(self, x, y, z):
        return _tricubic_dx(self, x, y, z)

    def deriv_y(self, x, y, z):
        return _tricubic_dy(self, x, y, z)

    def deriv_z(self, x, y, z):
        return _tricubic_dz(self, x, y, z)

    def deriv_xx(self, x, y, z):
        return _tricubic_dxx(self, x, y, z)

    def deriv_yy(self, x, y, z):
        return _tricubic_dyy(self, x, y, z)

    def deriv_zz(self, x, y, z):
        return _tricubic_dzz(self, x, y, z)

    def deriv_xy(self, x, y, z):
        return _tricubic_dxy(self, x, y, z)

    def deriv_xz(self, x, y, z):
        return _tricubic_dxz(self, x, y, z)

    def deriv_yz(self, x, y, z):
        return _tricubic_dyz(self, x, y, z)


class QuadcubicSplineJax(typing.NamedTuple):
    coeffs: typing.Any
    w0: typing.Any
    dw: typing.Any
    x0: typing.Any
    dx: typing.Any
    y0: typing.Any
    dy: typing.Any
    z0: typing.Any
    dz: typing.Any

    def eval(self, w, x, y, z):
        return _quadcubic_eval(self, w, x, y, z)

    def deriv_w(self, w, x, y, z):
        return _quadcubic_dw(self, w, x, y, z)

    def deriv_x(self, w, x, y, z):
        return _quadcubic_dx(self, w, x, y, z)

    def deriv_y(self, w, x, y, z):
        return _quadcubic_dy(self, w, x, y, z)

    def deriv_z(self, w, x, y, z):
        return _quadcubic_dz(self, w, x, y, z)

    def deriv_ww(self, w, x, y, z):
        return _quadcubic_dww(self, w, x, y, z)

    def deriv_xx(self, w, x, y, z):
        return _quadcubic_dxx(self, w, x, y, z)

    def deriv_yy(self, w, x, y, z):
        return _quadcubic_dyy(self, w, x, y, z)

    def deriv_zz(self, w, x, y, z):
        return _quadcubic_dzz(self, w, x, y, z)

    def deriv_wx(self, w, x, y, z):
        return _quadcubic_dwx(self, w, x, y, z)

    def deriv_wy(self, w, x, y, z):
        return _quadcubic_dwy(self, w, x, y, z)

    def deriv_wz(self, w, x, y, z):
        return _quadcubic_dwz(self, w, x, y, z)

    def deriv_xy(self, w, x, y, z):
        return _quadcubic_dxy(self, w, x, y, z)

    def deriv_xz(self, w, x, y, z):
        return _quadcubic_dxz(self, w, x, y, z)

    def deriv_yz(self, w, x, y, z):
        return _quadcubic_dyz(self, w, x, y, z)
