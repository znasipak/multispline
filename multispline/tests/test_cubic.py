import unittest
import numpy as np
from multispline.spline import CubicSplineUniformGrid, CubicSpline, available_boundary_conditions, cubic_spline_bc_dict

class TestCubicSpline(unittest.TestCase):
    def setUp(self):
        # Define a simple grid and function values for testing
        self.x0 = 0.0
        self.dx = 1.0
        self.x = np.linspace(self.x0, self.x0 + 5 * self.dx, 6)
        self.f = np.array([0.0, 1.0, 0.5, -1.0, -0.5, 0.0])
        
        # Boundary condition options to test
        self.boundary_conditions = available_boundary_conditions()
    
    def test_boundary_conditions(self):
        # Test that available boundary conditions match expected dictionary keys
        self.assertEqual(set(available_boundary_conditions()), set(cubic_spline_bc_dict.keys()))
    
    def test_initialization_with_uniform_grid(self):
        # Test correct initialization and boundary condition handling for CubicSplineUniformGrid
        for bc in self.boundary_conditions:
            spline = CubicSplineUniformGrid(self.x0, self.dx, self.f, bc)
            self.assertEqual(spline.x0, self.x0)
            self.assertEqual(spline.dx, self.dx)
            self.assertEqual(spline.nx, len(self.f) - 1)
            self.assertIn(bc, spline.available_boundary_conditions)

    def test_invalid_boundary_conditions(self):
        # Test invalid boundary condition handling
        with self.assertRaises(ValueError):
            CubicSplineUniformGrid(self.x0, self.dx, self.f, "invalid-bc")
    
    def test_coefficients_uniform_grid(self):
        # Test coefficients property for uniform grid spline
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f, "natural")
        coeffs = spline.coefficients
        self.assertEqual(coeffs.shape, (spline.nx, 4))  # Check shape of coefficients array
        # Test each coefficient for validity, i.e., it should be a float
        for i in range(spline.nx):
            for j in range(4):
                self.assertIsInstance(coeffs[i, j], float)

    def test_coeff_method(self):
        # Test coeff() method for correct coefficient retrieval
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f)
        for i in range(spline.nx):
            for mx in range(4):
                coeff = spline.coeff(i, mx)
                self.assertIsInstance(coeff, float)
    
    def test_eval_single_point(self):
        # Test eval() for a single point input
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f)
        x_test = 2.5
        result = spline.eval(x_test)
        self.assertIsInstance(result, float)

    def test_eval_array(self):
        # Test eval() for an array of points
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f)
        x_test = np.array([0.5, 1.5, 2.5])
        result = spline.eval(x_test)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x_test.shape)

    def test_deriv_single_point(self):
        # Test deriv() for a single point input
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f)
        x_test = 2.5
        result = spline.deriv(x_test)
        self.assertIsInstance(result, float)

    def test_deriv_array(self):
        # Test deriv() for an array of points
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f)
        x_test = np.array([0.5, 1.5, 2.5])
        result = spline.deriv(x_test)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x_test.shape)

    def test_deriv2_single_point(self):
        # Test deriv2() for a single point input
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f)
        x_test = 2.5
        result = spline.deriv2(x_test)
        self.assertIsInstance(result, float)

    def test_deriv2_array(self):
        # Test deriv2() for an array of points
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f)
        x_test = np.array([0.5, 1.5, 2.5])
        result = spline.deriv2(x_test)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x_test.shape)

    def test_call_method(self):
        # Test __call__ method
        spline = CubicSplineUniformGrid(self.x0, self.dx, self.f)
        x_test = 2.5
        self.assertEqual(spline(x_test), spline.eval(x_test))

    def test_initialization_with_non_uniform_grid(self):
        # Test correct initialization for non-uniform grid spline
        for bc in self.boundary_conditions:
            spline = CubicSpline(self.x, self.f, bc)
            self.assertEqual(spline.x0, self.x[0])
            self.assertEqual(spline.nx, len(self.f) - 1)
            self.assertIn(bc, spline.available_boundary_conditions)

    def test_non_uniform_grid_assertions(self):
        # Test non-uniform grid raises error
        x_non_uniform = np.array([0.0, 1.0, 2.5, 3.0, 4.0, 5.0])
        with self.assertRaises(AssertionError):
            CubicSpline(x_non_uniform, self.f)
    
    def test_shapes_match_assertion(self):
        # Test mismatch of x and f shapes raises error
        f_mismatched = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(AssertionError):
            CubicSpline(self.x, f_mismatched)

if __name__ == '__main__':
    unittest.main()
