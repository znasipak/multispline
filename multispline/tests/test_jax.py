"""Tests for the JAX-compatible spline evaluation layer.

The whole module is skipped if JAX is not installed. 64-bit mode is enabled at
import time (before any JAX computation) so results match the C++ backend to
double precision.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from multispline.spline import (  # noqa: E402
    CubicSpline,
    BicubicSpline,
    TricubicSpline,
    QuadcubicSpline,
)

TOL = 1e-10


# ---------------------------------------------------------------------------
# correctness: JAX (autodiff) matches the C++ backend
# ---------------------------------------------------------------------------

def test_cubic_matches_backend():
    x = np.linspace(0.5, 3.0, 21)
    f = np.sin(x)
    s = CubicSpline(x, f)
    js = s.to_jax()
    for xi in np.linspace(0.6, 2.9, 23):
        assert abs(s.eval(xi) - float(js.eval(xi))) < TOL
        assert abs(s.deriv(xi) - float(js.deriv(xi))) < TOL
        assert abs(s.deriv2(xi) - float(js.deriv2(xi))) < TOL


def test_bicubic_matches_backend():
    x = np.linspace(0, 2, 15)
    y = np.linspace(-1, 1, 13)
    F = np.sin(x)[:, None] * np.cos(y)[None, :]
    s = BicubicSpline(x, y, F)
    js = s.to_jax()
    rng = np.random.default_rng(0)
    methods = ["eval", "deriv_x", "deriv_y", "deriv_xx", "deriv_yy", "deriv_xy"]
    for _ in range(30):
        a, b = rng.uniform(0.1, 1.9), rng.uniform(-0.9, 0.9)
        for m in methods:
            assert abs(getattr(s, m)(a, b) - float(getattr(js, m)(a, b))) < TOL


def test_tricubic_matches_backend():
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 9)
    z = np.linspace(0, 1, 8)
    F = np.einsum("i,j,k->ijk", np.sin(x), np.cos(y), np.exp(z))
    s = TricubicSpline(x, y, z, F)
    js = s.to_jax()
    rng = np.random.default_rng(1)
    methods = ["eval", "deriv_x", "deriv_y", "deriv_z",
               "deriv_xx", "deriv_yy", "deriv_xy", "deriv_xz", "deriv_yz"]
    for _ in range(30):
        a, b, c = rng.uniform(0.05, 0.95, size=3)
        for m in methods:
            assert abs(getattr(s, m)(a, b, c) - float(getattr(js, m)(a, b, c))) < TOL


def test_tricubic_deriv_zz_matches_finite_difference():
    x = np.linspace(0, 1, 15)
    y = np.linspace(0, 1, 13)
    z = np.linspace(0, 1, 12)
    F = np.einsum("i,j,k->ijk", np.sin(2 * x), np.cos(y), np.exp(z))
    s = TricubicSpline(x, y, z, F)
    js = s.to_jax()
    a, b, c, h = 0.4, 0.5, 0.6, 1e-4
    fd = (s.eval(a, b, c + h) - 2 * s.eval(a, b, c) + s.eval(a, b, c - h)) / h ** 2
    assert abs(float(js.deriv_zz(a, b, c)) - fd) < 1e-4


def test_quadcubic_matches_backend():
    w = np.linspace(0, 1, 9)
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 7)
    z = np.linspace(0, 1, 6)
    F = np.einsum("h,i,j,k->hijk", np.sin(w), np.cos(x), np.exp(y), np.sin(2 * z))
    s = QuadcubicSpline(w, x, y, z, F)
    js = s.to_jax()
    rng = np.random.default_rng(2)
    methods = ["eval", "deriv_w", "deriv_x", "deriv_y", "deriv_z",
               "deriv_ww", "deriv_xx", "deriv_yy", "deriv_zz",
               "deriv_wx", "deriv_wy", "deriv_wz",
               "deriv_xy", "deriv_xz", "deriv_yz"]
    for _ in range(20):
        p = rng.uniform(0.05, 0.95, size=4)
        for m in methods:
            assert abs(getattr(s.base, m)(*p) - float(getattr(js, m)(*p))) < TOL


# ---------------------------------------------------------------------------
# the spline composes with the JAX transforms
# ---------------------------------------------------------------------------

def test_jit_over_user_function():
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 9)
    z = np.linspace(0, 1, 8)
    F = np.einsum("i,j,k->ijk", np.sin(x), np.cos(y), np.exp(z))
    s = TricubicSpline(x, y, z, F)
    js = s.to_jax()

    @jax.jit
    def integrate(spline, pts):
        return jnp.sum(jax.vmap(spline.eval)(pts[:, 0], pts[:, 1], pts[:, 2]))

    pts = jnp.asarray(np.random.default_rng(3).uniform(0.05, 0.95, size=(50, 3)))
    got = float(integrate(js, pts))
    expected = sum(s.eval(*p) for p in np.asarray(pts))
    assert abs(got - expected) < 1e-8


def test_vmap_batches_evaluation():
    x = np.linspace(0, 2, 15)
    y = np.linspace(-1, 1, 13)
    F = np.sin(x)[:, None] * np.cos(y)[None, :]
    s = BicubicSpline(x, y, F)
    js = s.to_jax()
    xs = jnp.linspace(0.2, 1.8, 32)
    ys = jnp.linspace(-0.8, 0.8, 32)
    batched = jax.vmap(js.eval)(xs, ys)
    expected = np.array([s.eval(float(a), float(b)) for a, b in zip(xs, ys)])
    assert np.allclose(np.asarray(batched), expected, atol=TOL)


def test_grad_equals_analytic_derivative():
    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 9)
    z = np.linspace(0, 1, 8)
    F = np.einsum("i,j,k->ijk", np.sin(x), np.cos(y), np.exp(z))
    s = TricubicSpline(x, y, z, F)
    js = s.to_jax()
    a, b, c = 0.3, 0.4, 0.5
    gx = jax.grad(js.eval, argnums=0)(a, b, c)
    assert abs(float(gx) - s.deriv_x(a, b, c)) < TOL


def test_to_jax_returns_expected_type():
    x = np.linspace(0, 1, 6)
    f = np.cos(x)
    js = CubicSpline(x, f).to_jax()
    assert type(js).__name__ == "CubicSplineJax"
