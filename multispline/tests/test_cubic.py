import pytest
import numpy as np
from multispline.spline import CubicSplineUniformGrid, CubicSpline, available_boundary_conditions, cubic_spline_bc_dict


@pytest.fixture
def uniform_grid():
    x0 = 0.0
    dx = 1.0
    x = np.linspace(x0, x0 + 5 * dx, 6)
    f = np.array([0.0, 1.0, 0.5, -1.0, -0.5, 0.0])
    return x0, dx, x, f


@pytest.fixture
def boundary_conditions():
    return available_boundary_conditions()


def test_boundary_conditions():
    assert set(available_boundary_conditions()) == set(cubic_spline_bc_dict.keys())


def test_initialization_with_uniform_grid(uniform_grid, boundary_conditions):
    x0, dx, x, f = uniform_grid
    for bc in boundary_conditions:
        spline = CubicSplineUniformGrid(x0, dx, f, bc)
        assert spline.x0 == x0
        assert spline.dx == dx
        assert spline.nx == len(f) - 1
        assert bc in spline.available_boundary_conditions


def test_invalid_boundary_conditions(uniform_grid):
    x0, dx, x, f = uniform_grid
    with pytest.raises(ValueError):
        CubicSplineUniformGrid(x0, dx, f, "invalid-bc")


def test_coefficients_uniform_grid(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f, "natural")
    coeffs = spline.coefficients
    assert coeffs.shape == (spline.nx, 4)
    for i in range(spline.nx):
        for j in range(4):
            c = coeffs[i, j]
            assert np.isscalar(c) or isinstance(c, (float, np.floating))


def test_coeff_method(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f)
    for i in range(spline.nx):
        for mx in range(4):
            coeff = spline.coeff(i, mx)
            assert np.isscalar(coeff) or isinstance(coeff, (float, np.floating))


def test_eval_single_point(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f)
    x_test = 2.5
    result = spline.eval(x_test)
    assert np.isscalar(result) or isinstance(result, (float, np.floating))


def test_eval_array(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f)
    x_test = np.array([0.5, 1.5, 2.5])
    result = spline.eval(x_test)
    assert isinstance(result, np.ndarray)
    assert result.shape == x_test.shape


def test_deriv_single_point(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f)
    x_test = 2.5
    result = spline.deriv(x_test)
    assert np.isscalar(result) or isinstance(result, (float, np.floating))


def test_deriv_array(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f)
    x_test = np.array([0.5, 1.5, 2.5])
    result = spline.deriv(x_test)
    assert isinstance(result, np.ndarray)
    assert result.shape == x_test.shape


def test_deriv2_single_point(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f)
    x_test = 2.5
    result = spline.deriv2(x_test)
    assert np.isscalar(result) or isinstance(result, (float, np.floating))


def test_deriv2_array(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f)
    x_test = np.array([0.5, 1.5, 2.5])
    result = spline.deriv2(x_test)
    assert isinstance(result, np.ndarray)
    assert result.shape == x_test.shape


def test_call_method(uniform_grid):
    x0, dx, x, f = uniform_grid
    spline = CubicSplineUniformGrid(x0, dx, f)
    x_test = 2.5
    assert spline(x_test) == spline.eval(x_test)


def test_initialization_with_non_uniform_grid(boundary_conditions):
    # Using a uniform x for the non-uniform-grid constructor to test initialization
    x = np.linspace(0.0, 5.0, 6)
    f = np.array([0.0, 1.0, 0.5, -1.0, -0.5, 0.0])
    for bc in boundary_conditions:
        spline = CubicSpline(x, f, bc)
        assert spline.x0 == x[0]
        assert spline.nx == len(f) - 1
        assert bc in spline.available_boundary_conditions


def test_non_uniform_grid_assertions():
    x_non_uniform = np.array([0.0, 1.0, 2.5, 3.0, 4.0, 5.0])
    f = np.array([0.0, 1.0, 0.5, -1.0, -0.5, 0.0])
    with pytest.raises(AssertionError):
        CubicSpline(x_non_uniform, f)


def test_shapes_match_assertion():
    x = np.linspace(0.0, 5.0, 6)
    f_mismatched = np.array([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError):
        CubicSpline(x, f_mismatched)

