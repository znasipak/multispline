import unittest
import numpy as np
from multispline.spline import TricubicSpline, available_boundary_conditions, cubic_spline_bc_dict

class TestTricubicSpline(unittest.TestCase):

    def setUp(self):
        # Setting up a simple 3D grid and corresponding function values
        self.x = np.linspace(0, 1, 5)  # grid in x
        self.y = np.linspace(0, 1, 5)  # grid in y
        self.z = np.linspace(0, 1, 5)  # grid in z
        self.f = np.random.rand(5, 5, 5)  # some function values on the grid
        
        # Set up instance with valid boundary condition
        self.spline = TricubicSpline(self.x, self.y, self.z, self.f, bc="natural")

    def test_boundary_conditions(self):
        # Test that available boundary conditions match expected dictionary keys
        self.assertEqual(set(available_boundary_conditions()), set(cubic_spline_bc_dict.keys()))
    
    def test_invalid_boundary_conditions(self):
        # Test invalid boundary condition
        with self.assertRaises(ValueError):
            TricubicSpline(self.x, self.y, self.z, self.f, bc="invalid-bc")

    def test_grid_spacing_assertions(self):
        # Test non-uniform grid spacing in x, y, z which should raise assertion errors
        non_uniform_x = np.array([0, 0.3, 0.6, 1.0, 1.5])
        with self.assertRaises(AssertionError):
            TricubicSpline(non_uniform_x, self.y, self.z, self.f)

    def test_eval_method(self):
        # Check if eval returns expected type and is callable
        x0, y0, z0 = 0.5, 0.5, 0.5
        result = self.spline.eval(x0, y0, z0)
        self.assertIsInstance(result, float)
        # Evaluating multiple points with array inputs
        points_x = np.array([0.2, 0.4, 0.8])
        points_y = np.array([0.2, 0.4, 0.8])
        points_z = np.array([0.2, 0.4, 0.8])
        result_array = self.spline.eval(points_x, points_y, points_z)
        self.assertIsInstance(result_array, np.ndarray)

    def test_partial_derivatives(self):
        # Test that partial derivatives work and return float for single points
        x0, y0, z0 = 0.5, 0.5, 0.5
        self.assertIsInstance(self.spline.deriv_x(x0, y0, z0), float)
        self.assertIsInstance(self.spline.deriv_y(x0, y0, z0), float)
        self.assertIsInstance(self.spline.deriv_z(x0, y0, z0), float)

    def test_second_partial_derivatives(self):
        # Test that second partial derivatives return correct types
        x0, y0, z0 = 0.5, 0.5, 0.5
        self.assertIsInstance(self.spline.deriv_xx(x0, y0, z0), float)
        self.assertIsInstance(self.spline.deriv_yy(x0, y0, z0), float)
        self.assertIsInstance(self.spline.deriv_zz(x0, y0, z0), float)

    def test_mixed_partial_derivatives(self):
        # Test mixed partial derivatives
        x0, y0, z0 = 0.5, 0.5, 0.5
        self.assertIsInstance(self.spline.deriv_xy(x0, y0, z0), float)
        self.assertIsInstance(self.spline.deriv_xz(x0, y0, z0), float)
        self.assertIsInstance(self.spline.deriv_yz(x0, y0, z0), float)

    def test_coefficients(self):
        # Test coefficients array
        coeffs = self.spline.coefficients
        self.assertIsInstance(coeffs, np.ndarray)
        expected_shape = (self.x.shape[0]-1, self.y.shape[0]-1, 64*(self.z.shape[0]-1))
        self.assertEqual(coeffs.shape, expected_shape)

    def test_coeff_method(self):
        # Check coeff method for specific coefficient
        coeff_value = self.spline.coeff(1, 1, 1, 2, 2, 2)
        self.assertIsInstance(coeff_value, float)

    def test_call_method(self):
        # Test the __call__ method that should be an alias to eval
        x0, y0, z0 = 0.5, 0.5, 0.5
        result_call = self.spline(x0, y0, z0)
        result_eval = self.spline.eval(x0, y0, z0)
        self.assertEqual(result_call, result_eval)

if __name__ == "__main__":
    unittest.main()
