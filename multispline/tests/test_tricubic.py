import pytest
import numpy as np
from multispline.spline import TricubicSpline, available_boundary_conditions, cubic_spline_bc_dict


@pytest.fixture
def grid3d():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    z = np.linspace(0, 1, 5)
    f = np.random.rand(5, 5, 5)
    return x, y, z, f


def test_boundary_conditions():
    assert set(available_boundary_conditions()) == set(cubic_spline_bc_dict.keys())


def test_invalid_boundary_conditions(grid3d):
    x, y, z, f = grid3d
    with pytest.raises(ValueError):
        TricubicSpline(x, y, z, f, bc="invalid-bc")


def test_grid_spacing_assertions(grid3d):
    x, y, z, f = grid3d
    non_uniform_x = np.array([0, 0.3, 0.6, 1.0, 1.5])
    with pytest.raises(AssertionError):
        TricubicSpline(non_uniform_x, y, z, f)


def test_eval_method(grid3d):
    x, y, z, f = grid3d
    spline = TricubicSpline(x, y, z, f, bc="natural")
    x0, y0, z0 = 0.5, 0.5, 0.5
    result = spline.eval(x0, y0, z0)
    assert np.isscalar(result) or isinstance(result, (float, np.floating))
    points_x = np.array([0.2, 0.4, 0.8])
    points_y = np.array([0.2, 0.4, 0.8])
    points_z = np.array([0.2, 0.4, 0.8])
    result_array = spline.eval(points_x, points_y, points_z)
    assert isinstance(result_array, np.ndarray)


def test_partial_derivatives(grid3d):
    x, y, z, f = grid3d
    spline = TricubicSpline(x, y, z, f, bc="natural")
    x0, y0, z0 = 0.5, 0.5, 0.5
    assert np.isscalar(spline.deriv_x(x0, y0, z0)) or isinstance(spline.deriv_x(x0, y0, z0), (float, np.floating))
    assert np.isscalar(spline.deriv_y(x0, y0, z0)) or isinstance(spline.deriv_y(x0, y0, z0), (float, np.floating))
    assert np.isscalar(spline.deriv_z(x0, y0, z0)) or isinstance(spline.deriv_z(x0, y0, z0), (float, np.floating))


def test_second_partial_derivatives(grid3d):
    x, y, z, f = grid3d
    spline = TricubicSpline(x, y, z, f, bc="natural")
    x0, y0, z0 = 0.5, 0.5, 0.5
    assert np.isscalar(spline.deriv_xx(x0, y0, z0)) or isinstance(spline.deriv_xx(x0, y0, z0), (float, np.floating))
    assert np.isscalar(spline.deriv_yy(x0, y0, z0)) or isinstance(spline.deriv_yy(x0, y0, z0), (float, np.floating))
    assert np.isscalar(spline.deriv_zz(x0, y0, z0)) or isinstance(spline.deriv_zz(x0, y0, z0), (float, np.floating))


def test_mixed_partial_derivatives(grid3d):
    x, y, z, f = grid3d
    spline = TricubicSpline(x, y, z, f, bc="natural")
    x0, y0, z0 = 0.5, 0.5, 0.5
    assert np.isscalar(spline.deriv_xy(x0, y0, z0)) or isinstance(spline.deriv_xy(x0, y0, z0), (float, np.floating))
    assert np.isscalar(spline.deriv_xz(x0, y0, z0)) or isinstance(spline.deriv_xz(x0, y0, z0), (float, np.floating))
    assert np.isscalar(spline.deriv_yz(x0, y0, z0)) or isinstance(spline.deriv_yz(x0, y0, z0), (float, np.floating))


def test_coefficients(grid3d):
    x, y, z, f = grid3d
    spline = TricubicSpline(x, y, z, f, bc="natural")
    coeffs = spline.coefficients
    assert isinstance(coeffs, np.ndarray)
    expected_shape = (x.shape[0]-1, y.shape[0]-1, 64*(z.shape[0]-1))
    assert coeffs.shape == expected_shape


def test_coeff_method(grid3d):
    x, y, z, f = grid3d
    spline = TricubicSpline(x, y, z, f, bc="natural")
    coeff_value = spline.coeff(1, 1, 1, 2, 2, 2)
    assert np.isscalar(coeff_value) or isinstance(coeff_value, (float, np.floating))


def test_call_method(grid3d):
    x, y, z, f = grid3d
    spline = TricubicSpline(x, y, z, f, bc="natural")
    x0, y0, z0 = 0.5, 0.5, 0.5
    result_call = spline(x0, y0, z0)
    result_eval = spline.eval(x0, y0, z0)
    assert result_call == result_eval

