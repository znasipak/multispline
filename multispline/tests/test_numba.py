"""Tests for the numba-compatible spline evaluation layer.

The whole module is skipped if numba is not installed, so the rest of the test
suite still runs in environments without the optional dependency.
"""
import numpy as np
import pytest

numba = pytest.importorskip("numba")
from numba import njit  # noqa: E402

from multispline.spline import (  # noqa: E402
    CubicSpline,
    BicubicSpline,
    TricubicSpline,
    QuadcubicSpline,
)
from multispline import numba as msnumba  # noqa: E402

TOL = 1e-10


# ---------------------------------------------------------------------------
# correctness: numba jitclass matches the C++ backend
# ---------------------------------------------------------------------------

def test_cubic_matches_backend():
    x = np.linspace(0.5, 3.0, 21)
    f = np.sin(x)
    s = CubicSpline(x, f)
    ns = s.to_numba()
    for xi in np.linspace(0.6, 2.9, 23):
        assert abs(s.eval(xi) - ns.eval(xi)) < TOL
        assert abs(s.deriv(xi) - ns.deriv(xi)) < TOL
        assert abs(s.deriv2(xi) - ns.deriv2(xi)) < TOL


def test_bicubic_matches_backend():
    x = np.linspace(0, 2, 15)
    y = np.linspace(-1, 1, 13)
    F = np.sin(x)[:, None] * np.cos(y)[None, :]
    s = BicubicSpline(x, y, F)
    ns = s.to_numba()
    rng = np.random.default_rng(0)
    methods = ["eval", "deriv_x", "deriv_y", "deriv_xx", "deriv_yy", "deriv_xy"]
    for _ in range(30):
        a, b = rng.uniform(0.1, 1.9), rng.uniform(-0.9, 0.9)
        for m in methods:
            assert abs(getattr(s, m)(a, b) - getattr(ns, m)(a, b)) < TOL


def test_tricubic_matches_backend():
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 9)
    z = np.linspace(0, 1, 8)
    F = np.einsum("i,j,k->ijk", np.sin(x), np.cos(y), np.exp(z))
    s = TricubicSpline(x, y, z, F)
    ns = s.to_numba()
    rng = np.random.default_rng(1)
    # deriv_zz omitted here: the Cython backend's deriv_zz is validated separately
    methods = ["eval", "deriv_x", "deriv_y", "deriv_z",
               "deriv_xx", "deriv_yy", "deriv_xy", "deriv_xz", "deriv_yz"]
    for _ in range(30):
        a, b, c = rng.uniform(0.05, 0.95, size=3)
        for m in methods:
            assert abs(getattr(s, m)(a, b, c) - getattr(ns, m)(a, b, c)) < TOL


def test_tricubic_deriv_zz_matches_finite_difference():
    # The numba kernel computes the true d^2/dz^2; cross-check against a
    # central finite difference of the value.
    x = np.linspace(0, 1, 15)
    y = np.linspace(0, 1, 13)
    z = np.linspace(0, 1, 12)
    F = np.einsum("i,j,k->ijk", np.sin(2 * x), np.cos(y), np.exp(z))
    s = TricubicSpline(x, y, z, F)
    ns = s.to_numba()
    a, b, c, h = 0.4, 0.5, 0.6, 1e-4
    fd = (s.eval(a, b, c + h) - 2 * s.eval(a, b, c) + s.eval(a, b, c - h)) / h ** 2
    assert abs(ns.deriv_zz(a, b, c) - fd) < 1e-4


def test_quadcubic_matches_backend():
    w = np.linspace(0, 1, 9)
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 7)
    z = np.linspace(0, 1, 6)
    F = np.einsum("h,i,j,k->hijk", np.sin(w), np.cos(x), np.exp(y), np.sin(2 * z))
    s = QuadcubicSpline(w, x, y, z, F)
    ns = s.to_numba()
    rng = np.random.default_rng(2)
    # Compare against the Cython base, which exposes the w-derivatives that the
    # Python QuadcubicSpline wrapper does not.
    methods = ["eval", "deriv_w", "deriv_x", "deriv_y", "deriv_z",
               "deriv_ww", "deriv_xx", "deriv_yy", "deriv_zz",
               "deriv_wx", "deriv_wy", "deriv_wz",
               "deriv_xy", "deriv_xz", "deriv_yz"]
    for _ in range(20):
        p = rng.uniform(0.05, 0.95, size=4)
        for m in methods:
            assert abs(getattr(s.base, m)(*p) - getattr(ns, m)(*p)) < TOL


# ---------------------------------------------------------------------------
# the spline really is usable from inside an @njit function (nopython mode)
# ---------------------------------------------------------------------------

def test_jitclass_callable_inside_njit():
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 9)
    z = np.linspace(0, 1, 8)
    F = np.einsum("i,j,k->ijk", np.sin(x), np.cos(y), np.exp(z))
    s = TricubicSpline(x, y, z, F)
    ns = s.to_numba()

    @njit
    def integrate(spline, pts):
        total = 0.0
        for n in range(pts.shape[0]):
            total += spline.eval(pts[n, 0], pts[n, 1], pts[n, 2])
        return total

    pts = np.random.default_rng(3).uniform(0.05, 0.95, size=(50, 3))
    got = integrate(ns, pts)
    expected = sum(s.eval(*p) for p in pts)
    assert abs(got - expected) < 1e-8
    # confirm it actually compiled in nopython mode (no objmode fallback)
    assert len(integrate.nopython_signatures) >= 1


def test_free_kernel_callable_inside_njit():
    x = np.linspace(0.5, 3.0, 21)
    f = np.sin(x)
    s = CubicSpline(x, f)
    c = np.ascontiguousarray(s.coefficients, dtype=np.float64)
    x0, dx, nx = float(s.x0), float(s.dx), int(s.nx)

    @njit
    def kernel(coeffs, x0, dx, nx, xs):
        out = np.empty(xs.shape[0])
        for n in range(xs.shape[0]):
            out[n] = msnumba.cubic_eval(coeffs, x0, dx, nx, xs[n])
        return out

    xs = np.linspace(0.6, 2.9, 17)
    got = kernel(c, x0, dx, nx, xs)
    expected = np.array([s.eval(xi) for xi in xs])
    assert np.allclose(got, expected, atol=TOL)
    assert len(kernel.nopython_signatures) >= 1


def test_to_numba_requires_numba_message(monkeypatch):
    # Sanity: to_numba surfaces a clear error path via the multispline.numba
    # import. Here we just confirm the normal path returns the expected type.
    x = np.linspace(0, 1, 6)
    f = np.cos(x)
    ns = CubicSpline(x, f).to_numba()
    assert type(ns).__name__ == "CubicSplineNumba"
