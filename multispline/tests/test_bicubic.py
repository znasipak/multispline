import pytest
import numpy as np
from multispline.spline import BicubicSpline, available_boundary_conditions, cubic_spline_bc_dict


@pytest.fixture
def grid():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    f = np.array([[np.sin(xi) * np.cos(yj) for yj in y] for xi in x])
    return x, y, f


def test_initialization_valid(grid):
    x, y, f = grid
    # If construction raises, the test will fail
    BicubicSpline(x, y, f)


def test_initialization_invalid_shape(grid):
    x, y, _ = grid
    f_invalid = np.random.rand(5, 6)
    with pytest.raises(AssertionError):
        BicubicSpline(x, y, f_invalid)


def test_invalid_boundary_condition(grid):
    x, y, f = grid
    with pytest.raises(ValueError):
        BicubicSpline(x, y, f, bc="invalid_bc")


def test_boundary_conditions():
    assert set(available_boundary_conditions()) == set(cubic_spline_bc_dict.keys())


def test_eval_scalar(grid):
    x, y, f = grid
    spline = BicubicSpline(x, y, f)
    result = spline.eval(0.5, 0.5)
    assert np.isscalar(result) or isinstance(result, (float, np.floating))


def test_eval_array(grid):
    x, y, f = grid
    spline = BicubicSpline(x, y, f)
    x_test = np.array([0.2, 0.4, 0.6])
    y_test = np.array([0.2, 0.4, 0.6])
    result = spline.eval(x_test, y_test)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_partial_derivatives(grid):
    x, y, f = grid
    spline = BicubicSpline(x, y, f)
    deriv_x = spline.deriv_x(0.5, 0.5)
    deriv_y = spline.deriv_y(0.5, 0.5)
    assert np.isscalar(deriv_x) or isinstance(deriv_x, (float, np.floating))
    assert np.isscalar(deriv_y) or isinstance(deriv_y, (float, np.floating))


def test_second_partial_derivatives(grid):
    x, y, f = grid
    spline = BicubicSpline(x, y, f)
    deriv_xx = spline.deriv_xx(0.5, 0.5)
    deriv_yy = spline.deriv_yy(0.5, 0.5)
    deriv_xy = spline.deriv_xy(0.5, 0.5)
    assert np.isscalar(deriv_xx) or isinstance(deriv_xx, (float, np.floating))
    assert np.isscalar(deriv_yy) or isinstance(deriv_yy, (float, np.floating))
    assert np.isscalar(deriv_xy) or isinstance(deriv_xy, (float, np.floating))


def test_coefficients_structure(grid):
    x, y, f = grid
    spline = BicubicSpline(x, y, f)
    coeffs = spline.coefficients
    assert tuple(coeffs.shape) == (4, 4, 4, 4)


def test_coeff_method(grid):
    x, y, f = grid
    spline = BicubicSpline(x, y, f)
    coeff_val = spline.coeff(0, 0, 1, 1)
    assert np.isscalar(coeff_val) or isinstance(coeff_val, (float, np.floating))


def test_call_method(grid):
    x, y, f = grid
    spline = BicubicSpline(x, y, f)
    result = spline(0.3, 0.3)
    assert np.isscalar(result) or isinstance(result, (float, np.floating))

