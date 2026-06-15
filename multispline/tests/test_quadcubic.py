import pytest
import numpy as np
from multispline.spline import QuadcubicSpline, available_boundary_conditions, cubic_spline_bc_dict


@pytest.fixture
def grid4d():
    w = np.linspace(0, 1, 4)
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 4)
    z = np.linspace(0, 1, 4)
    # 4D sample values on the grid (nw, nx, ny, nz)
    f = np.random.rand(w.shape[0], x.shape[0], y.shape[0], z.shape[0])
    return w, x, y, z, f


def test_boundary_conditions():
    assert set(available_boundary_conditions()) == set(cubic_spline_bc_dict.keys())


def test_invalid_boundary_conditions(grid4d):
    w, x, y, z, f = grid4d
    with pytest.raises(ValueError):
        QuadcubicSpline(w, x, y, z, f, bc="invalid-bc")


def test_grid_spacing_assertions(grid4d):
    w, x, y, z, f = grid4d
    non_uniform_w = np.array([0.0, 0.2, 0.55, 1.0])
    with pytest.raises(AssertionError):
        QuadcubicSpline(non_uniform_w, x, y, z, f)


def test_eval_and_broadcasting(grid4d):
    w, x, y, z, f = grid4d
    spline = QuadcubicSpline(w, x, y, z, f, bc="natural")
    # scalar evaluation
    result = spline.eval(0.3, 0.3, 0.3, 0.3)
    assert np.isscalar(result) or isinstance(result, (float, np.floating))

    # array broadcasting
    wa = np.array([0.2, 0.4])
    xa = np.array([0.2, 0.4])
    ya = np.array([0.2, 0.4])
    za = np.array([0.2, 0.4])
    out = spline.eval(wa, xa, ya, za)
    assert isinstance(out, np.ndarray)
    assert out.shape == wa.shape


def test_partial_derivatives(grid4d):
    w, x, y, z, f = grid4d
    spline = QuadcubicSpline(w, x, y, z, f, bc="natural")
    p = (0.5, 0.5, 0.5, 0.5)
    assert np.isscalar(spline.deriv_w(*p)) or isinstance(spline.deriv_w(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_x(*p)) or isinstance(spline.deriv_x(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_y(*p)) or isinstance(spline.deriv_y(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_z(*p)) or isinstance(spline.deriv_z(*p), (float, np.floating))


def test_second_and_mixed_derivatives(grid4d):
    w, x, y, z, f = grid4d
    spline = QuadcubicSpline(w, x, y, z, f, bc="natural")
    p = (0.5, 0.5, 0.5, 0.5)
    assert np.isscalar(spline.deriv_ww(*p)) or isinstance(spline.deriv_ww(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_xx(*p)) or isinstance(spline.deriv_xx(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_yy(*p)) or isinstance(spline.deriv_yy(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_zz(*p)) or isinstance(spline.deriv_zz(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_wx(*p)) or isinstance(spline.deriv_wx(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_wy(*p)) or isinstance(spline.deriv_wy(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_wz(*p)) or isinstance(spline.deriv_wz(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_xy(*p)) or isinstance(spline.deriv_xy(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_xz(*p)) or isinstance(spline.deriv_xz(*p), (float, np.floating))
    assert np.isscalar(spline.deriv_yz(*p)) or isinstance(spline.deriv_yz(*p), (float, np.floating))


def test_coefficients_and_coeff_method(grid4d):
    w, x, y, z, f = grid4d
    spline = QuadcubicSpline(w, x, y, z, f, bc="natural")
    coeffs = spline.coefficients
    assert isinstance(coeffs, np.ndarray)
    expected_shape = (w.shape[0]-1, x.shape[0]-1, y.shape[0]-1, 256*(z.shape[0]-1))
    assert coeffs.shape == expected_shape

    # pick a coefficient via coeff() and ensure it's numeric
    val = spline.coeff(0, 0, 0, 0, 1, 1, 1, 1)
    assert np.isscalar(val) or isinstance(val, (float, np.floating))


def test_call_alias(grid4d):
    w, x, y, z, f = grid4d
    spline = QuadcubicSpline(w, x, y, z, f, bc="natural")
    res_call = spline(0.25, 0.25, 0.25, 0.25)
    res_eval = spline.eval(0.25, 0.25, 0.25, 0.25)
    assert res_call == res_eval


def test_w_derivatives_broadcasting(grid4d):
    w, x, y, z, f = grid4d
    spline = QuadcubicSpline(w, x, y, z, f, bc="natural")
    wa = np.array([0.2, 0.4])
    xa = np.array([0.2, 0.4])
    ya = np.array([0.2, 0.4])
    za = np.array([0.2, 0.4])
    for method in (spline.deriv_w, spline.deriv_ww,
                   spline.deriv_wx, spline.deriv_wy, spline.deriv_wz):
        out = method(wa, xa, ya, za)
        assert isinstance(out, np.ndarray)
        assert out.shape == wa.shape
