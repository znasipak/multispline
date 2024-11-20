import unittest
import numpy as np
from multispline.spline import BicubicSpline, available_boundary_conditions, cubic_spline_bc_dict

class TestBicubicSpline(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 1, 5)
        self.y = np.linspace(0, 1, 5)
        self.f = np.array([[np.sin(xi) * np.cos(yj) for yj in self.y] for xi in self.x])

    def test_initialization_valid(self):
        # Test with valid input data
        try:
            spline = BicubicSpline(self.x, self.y, self.f)
            self.assertTrue(True)  # If no exception is raised, initialization is valid
        except Exception:
            self.fail("Initialization with valid data raised an exception unexpectedly.")

    def test_initialization_invalid_shape(self):
        # Test with invalid shape for `f`
        f_invalid = np.random.rand(5, 6)
        with self.assertRaises(AssertionError):
            BicubicSpline(self.x, self.y, f_invalid)

    def test_invalid_boundary_condition(self):
        # Test with an invalid boundary condition
        with self.assertRaises(ValueError):
            BicubicSpline(self.x, self.y, self.f, bc="invalid_bc")

    def test_boundary_conditions(self):
        # Test that available boundary conditions match expected dictionary keys
        self.assertEqual(set(available_boundary_conditions()), set(cubic_spline_bc_dict.keys()))
    
    def test_eval_scalar(self):
        # Test evaluation at a single point
        spline = BicubicSpline(self.x, self.y, self.f)
        result = spline.eval(0.5, 0.5)
        self.assertIsInstance(result, float)  # Assuming CyBicubicSpline.eval returns a scalar for single points

    def test_eval_array(self):
        # Test evaluation at arrays of points (testing numpy broadcasting)
        spline = BicubicSpline(self.x, self.y, self.f)
        x_test = np.array([0.2, 0.4, 0.6])
        y_test = np.array([0.2, 0.4, 0.6])
        result = spline.eval(x_test, y_test)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))

    def test_partial_derivatives(self):
        # Test first partial derivatives at a scalar point
        spline = BicubicSpline(self.x, self.y, self.f)
        x, y = 0.5, 0.5
        deriv_x = spline.deriv_x(x, y)
        deriv_y = spline.deriv_y(x, y)
        self.assertIsInstance(deriv_x, float)
        self.assertIsInstance(deriv_y, float)

    def test_second_partial_derivatives(self):
        # Test second partial derivatives at a scalar point
        spline = BicubicSpline(self.x, self.y, self.f)
        x, y = 0.5, 0.5
        deriv_xx = spline.deriv_xx(x, y)
        deriv_yy = spline.deriv_yy(x, y)
        deriv_xy = spline.deriv_xy(x, y)
        self.assertIsInstance(deriv_xx, float)
        self.assertIsInstance(deriv_yy, float)
        self.assertIsInstance(deriv_xy, float)

    def test_coefficients_structure(self):
        # Verify coefficients are structured as a 4D array with the expected shape
        spline = BicubicSpline(self.x, self.y, self.f)
        coeffs = spline.coefficients
        self.assertEqual(coeffs.shape, (4, 4, 4, 4))

    def test_coeff_method(self):
        # Test accessing individual coefficients using coeff method
        spline = BicubicSpline(self.x, self.y, self.f)
        coeff_val = spline.coeff(0, 0, 1, 1)
        self.assertIsInstance(coeff_val, float)

    def test_call_method(self):
        # Test calling the instance as a function
        spline = BicubicSpline(self.x, self.y, self.f)
        result = spline(0.3, 0.3)
        self.assertIsInstance(result, float)

# Run the tests
if __name__ == '__main__':
    unittest.main()
