import numpy as np
from multispline.spline import QuadcubicSpline

def test_function_4d(w, x, y, z):
    return (np.sin(x) * np.cos(4*y) + np.i0(3.2*z)*np.tanh(2.1*w))

NW = 14
NX = 15
NY = 10
NZ = 15
sample_points_w = np.linspace(-1, 1, NW)
sample_points_x = np.linspace(0, 5, NX)
sample_points_y = np.linspace(0, 5, NY)
sample_points_z = np.linspace(-2, 2, NZ)
sample_points_grid = np.meshgrid(sample_points_w, sample_points_x, sample_points_y, sample_points_z, indexing='ij')
sample_values_f = test_function_4d(*sample_points_grid)

qspl = QuadcubicSpline(sample_points_w, sample_points_x, sample_points_y, sample_points_z, sample_values_f)