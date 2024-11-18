# Tutorial

Import basic packages


```python
import numpy as np
import matplotlib.pyplot as plt
```

## Cubic Spline (1D)

The class `CubicSpline` provides a method for interpolating data sampled on a uniform one-dimensional grid.


```python
from multispline.spline import CubicSpline
```

As a first example we interpolate an oscillatory function, sampled at 100 equidistant points, using `CubicSpline`


```python
def test_function(x):
    return np.sin(x) + 0.1 * np.cos(10*x)

sample_points = np.linspace(0, 5, 100)
sample_values = test_function(sample_points)
```

Constructing an interpolating cubic spline is then as simple as initializing the `CubicSpline` class


```python
cspl = CubicSpline(sample_points, sample_values)
```

We can evaluate the spline to visually check how well the spline approximates our test function


```python
plt.plot(sample_points, sample_values, '.', label='original')
plt.plot(sample_points, cspl(sample_points), label='spline')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
```


    
![png](tutorial_files/tutorial_11_0.png)
    


To test the accuracy of the spline, we can also evaluate it at values that differ from the sample values


```python
test_points = np.linspace(0, 5, 66)
test_values = test_function(test_points)
```


```python
plt.plot(test_points, np.abs(test_values - cspl(test_points)), '.')
# plt.legend()
plt.xlabel('x')
plt.yscale('log')
plt.ylabel('Absolute error')
plt.show()
```


    
![png](tutorial_files/tutorial_14_0.png)
    


Cubic splines are not fully constrained by the sample data, but require a choice of boundary conditions. We can check the different types of boundary conditions offered by `multispline` by calling `available_boundary_conditions()`


```python
from multispline.spline import available_boundary_conditions
available_boundary_conditions()
```




    ['natural', 'not-a-knot', 'clamped', 'E(3)']



The default choice is known as the "E(3)" boundary condition. Other boundary conditions may be specified using the optional `bc` argument when instantiating `CubicSpline`


```python
cspl_natural = CubicSpline(sample_points, sample_values, bc='natural')
```

Notice that this new spline will not exactly agree with the original spline


```python
print(cspl_natural(0.001) - cspl(0.001))
```




    -0.00014990567110854947



Different boundary conditions may be better at approximating the behavior of the test function near the boundary. However, which boundary condition works best depends on the problem and the number of sample points. Below, we demonstrate how the "E(3)" boundary condition most accurately approximates the test function near the boundary, though the "not-a-knot" algorithm provides similarly well and more accurately approximates the test function at some points near the boundary.


```python
cspl_bcs = {}
for bc in available_boundary_conditions():
    cspl_bcs[bc] = CubicSpline(sample_points, sample_values, bc=bc)

for bc in available_boundary_conditions():
    plt.plot(test_points[1:-1], np.abs(cspl_bcs[bc](test_points[1:-1]) - test_function(test_points[1:-1])), label=bc)
plt.legend()
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Absolute error')
plt.show()
```


    
![png](tutorial_files/tutorial_22_0.png)
    


We can also evaluate the first and second derivatives of the spline through the methods `deriv` and `deriv2`


```python
print(cspl.deriv(0.4325))
print(cspl.deriv2(0.4325))
```

    1.8338073517167228
    3.3129057480026125


And we can directly access the spline coefficients through the class property `coefficients`---which returns a full array of coefficients---or the method `coeff` which returns a single coefficient for a given interval and polynomial power. (See documentation for more information.)


```python
print(cspl.coefficients)
print(cspl.coefficients[43, 3])
print(cspl.coeff(43, 3))
```

    [[ 1.00000000e-01  5.08213538e-02 -1.35749987e-02  7.52230981e-04]
     [ 1.37998586e-01  2.59280493e-02 -1.13183057e-02  1.40760863e-03]
     [ 1.54015938e-01  7.51426371e-03 -7.09547987e-03  2.06298627e-03]
     [ 1.56497708e-01 -4.87737213e-04 -9.06521058e-04  2.10237484e-03]
     [ 1.57205825e-01  4.00634520e-03  5.40060347e-03  1.63719121e-03]
     [ 1.68249965e-01  1.97191258e-02  1.03121771e-02  7.51166869e-04]
     [ 1.99032435e-01  4.25969806e-02  1.25656777e-02 -3.25688390e-04]
     [ 2.53869404e-01  6.67512708e-02  1.15886125e-02 -1.32673577e-03]
     [ 3.30882552e-01  8.59482885e-02  7.60840521e-03 -2.00129622e-03]
     [ 4.22437949e-01  9.51612103e-02  1.60451654e-03 -2.18099410e-03]
     [ 5.17022682e-01  9.18272611e-02 -4.93846575e-03 -1.82079632e-03]
     [ 6.02090681e-01  7.64879406e-02 -1.04008547e-02 -1.01052660e-03]
     [ 6.67167240e-01  5.26546514e-02 -1.34324345e-02  4.76353556e-05]
     [ 7.06437093e-01  2.59326884e-02 -1.32895284e-02  1.08961839e-03]
     [ 7.20169871e-01  2.62248670e-03 -1.00206733e-02  1.85540324e-03]
     [ 7.14627088e-01 -1.18526501e-02 -4.45446355e-03  2.15394753e-03]
     [ 7.00473922e-01 -1.42997346e-02  2.00737905e-03  1.91088896e-03]
     [ 6.90092455e-01 -4.55230966e-03  7.74004592e-03  1.18711298e-03]
     [ 6.94467304e-01  1.44891211e-02  1.13013849e-02  1.63549267e-04]
     [ 7.20421359e-01  3.75825387e-02  1.17920327e-02 -9.04006958e-04]
     [ 7.68891924e-01  5.84545831e-02  9.08001179e-03 -1.74876750e-03]
     [ 8.34677751e-01  7.13683042e-02  3.83370929e-03 -2.15956874e-03]
     [ 9.07720196e-01  7.25570165e-02 -2.64499694e-03 -2.03359977e-03]
     [ 9.75598616e-01  6.11662233e-02 -8.74579626e-03 -1.40207485e-03]
     [ 1.02661697e+00  3.94684063e-02 -1.29520208e-02 -4.22439799e-04]
     [ 1.05271091e+00  1.22970453e-02 -1.42193402e-02  6.60941520e-04]
     [ 1.05144956e+00 -1.41588105e-02 -1.22365156e-02  1.57780426e-03]
     [ 1.02663204e+00 -3.38984290e-02 -7.50310284e-03  2.09946704e-03]
     [ 9.87329974e-01 -4.26062336e-02 -1.20470173e-03  2.09593296e-03]
     [ 9.45614971e-01 -3.87278382e-02  5.08309716e-03  1.56834923e-03]
     [ 9.13538579e-01 -2.38565962e-02  9.78814484e-03  6.48719943e-04]
     [ 9.00118848e-01 -2.33414666e-03  1.17343047e-02 -4.33055930e-04]
     [ 9.09085950e-01  1.98352949e-02  1.04351369e-02 -1.40659104e-03]
     [ 9.37949791e-01  3.64857955e-02  6.21536376e-03 -2.02852600e-03]
     [ 9.78622424e-01  4.28309450e-02  1.29785753e-04 -2.14329694e-03]
     [ 1.01943986e+00  3.66606257e-02 -6.30010507e-03 -1.72198041e-03]
     [ 1.04807840e+00  1.88944743e-02 -1.14660463e-02 -8.69516289e-04]
     [ 1.05463731e+00 -6.64616713e-03 -1.40745952e-02  2.01495049e-04]
     [ 1.03411804e+00 -3.41908723e-02 -1.34701100e-02  1.22387836e-03]
     [ 9.87680939e-01 -5.74594573e-02 -9.79847495e-03  1.94259651e-03]
     [ 9.22365603e-01 -7.12286176e-02 -3.97068541e-03  2.17843261e-03]
     [ 8.49344733e-01 -7.26346906e-02  2.56461243e-03  1.87273980e-03]
     [ 7.81147394e-01 -6.18872464e-02  8.18283182e-03  1.10208475e-03]
     [ 7.28545064e-01 -4.22153285e-02  1.14890861e-02  5.91283978e-05]
     [ 6.97877950e-01 -1.90597712e-02  1.16664713e-02 -9.95482075e-04]
     [ 6.89489168e-01  1.28672510e-03  8.68002503e-03 -1.79819739e-03]
     [ 6.97657721e-01  1.32521830e-02  3.28543285e-03 -2.14837495e-03]
     [ 7.12046962e-01  1.33779238e-02 -3.15969199e-03 -1.95837979e-03]
     [ 7.20306814e-01  1.18340050e-03 -9.03483135e-03 -1.27546752e-03]
     [ 7.11179916e-01 -2.07126648e-02 -1.28612339e-02 -2.69985114e-04]
     [ 6.77336032e-01 -4.72450880e-02 -1.36711893e-02  8.07164329e-04]
     [ 6.17226919e-01 -7.21659735e-02 -1.12496963e-02  1.68717159e-03]
     [ 5.35498421e-01 -8.96038513e-02 -6.18818150e-03  2.15044262e-03]
     [ 4.41856831e-01 -9.55288864e-02  2.63146362e-04  2.08143075e-03]
     [ 3.48672521e-01 -8.87583015e-02  6.50743861e-03  1.49748838e-03]
     [ 2.67919147e-01 -7.12509591e-02  1.09999037e-02  5.44533763e-04]
     [ 2.08212625e-01 -4.76175503e-02  1.26335050e-02 -5.39385033e-04]
     [ 1.72689195e-01 -2.39686954e-02  1.10153499e-02 -1.48353097e-03]
     [ 1.58252318e-01 -6.38858844e-03  6.56475702e-03 -2.05208125e-03]
     [ 1.56376406e-01  5.84681865e-04  4.08513279e-04 -2.10301244e-03]
     [ 1.55266588e-01 -4.90732890e-03 -5.90052405e-03 -1.62356386e-03]
     [ 1.42835172e-01 -2.15790686e-02 -1.07712156e-02 -7.33424013e-04]
     [ 1.09751463e-01 -4.53217719e-02 -1.29714877e-02  3.45155531e-04]
     [ 5.18033594e-02 -7.02292806e-02 -1.19360211e-02  1.34285630e-03]
     [-2.90190860e-02 -9.00727539e-02 -7.90745218e-03  2.01054170e-03]
     [-1.24988750e-01 -9.98560331e-02 -1.87582708e-03  2.18146647e-03]
     [-2.24539144e-01 -9.70632879e-02  4.66857234e-03  1.81291297e-03]
     [-3.15120947e-01 -8.22874043e-02  1.01073113e-02  9.96857917e-04]
     [-3.86304182e-01 -5.90822080e-02  1.30978850e-02 -6.29941057e-05]
     [-4.32351499e-01 -3.30754203e-02  1.29089027e-02 -1.10207547e-03]
     [-4.53620092e-01 -1.05638413e-02  9.60267628e-03 -1.86101784e-03]
     [-4.56442275e-01  3.05845771e-03  4.01962277e-03 -2.15041618e-03]
     [-4.51514611e-01  4.64645471e-03 -2.43162578e-03 -1.89812295e-03]
     [-4.51197905e-01 -5.91116570e-03 -8.12599463e-03 -1.16726300e-03]
     [-4.66402328e-01 -2.56649440e-02 -1.16277836e-02 -1.40471011e-04]
     [-5.03835527e-01 -4.93419243e-02 -1.20491967e-02  9.25712748e-04]
     [-5.64300935e-01 -7.06631794e-02 -9.27205843e-03  1.76490087e-03]
     [-6.42471272e-01 -8.39125936e-02 -3.97735582e-03  2.16737615e-03]
     [-7.28193845e-01 -8.53651768e-02  2.52477263e-03  2.03245838e-03]
     [-8.09001791e-01 -7.42182564e-02  8.62214778e-03  1.39364469e-03]
     [-8.73204255e-01 -5.27930268e-02  1.28030818e-02  4.10245735e-04]
     [-9.12783954e-01 -2.59561259e-02  1.40338190e-02 -6.72393566e-04]
     [-9.25378654e-01  9.43314705e-05  1.20166383e-02 -1.58415616e-03]
     [-9.14851841e-01  1.93751397e-02  7.26416986e-03 -2.09760050e-03]
     [-8.90310132e-01  2.76106779e-02  9.71368362e-04 -2.08475209e-03]
     [-8.63812838e-01  2.32991584e-02 -5.28288790e-03 -1.54905811e-03]
     [-8.47345625e-01  8.08620821e-03 -9.93006224e-03 -6.24526163e-04]
     [-8.49814005e-01 -1.36474948e-02 -1.18036407e-02  4.57738516e-04]
     [-8.74807402e-01 -3.58815607e-02 -1.04304252e-02  1.42723891e-03]
     [-9.19692149e-01 -5.24606943e-02 -6.14870844e-03  2.04163641e-03]
     [-9.76259916e-01 -5.86332019e-02 -2.37992136e-05  2.14723937e-03]
     [-1.03276968e+00 -5.22390823e-02  6.41791889e-03  1.71747019e-03]
     [-1.07687337e+00 -3.42508339e-02  1.15703295e-02  8.59167991e-04]
     [-1.09869471e+00 -8.53267103e-03  1.41478334e-02 -2.12840285e-04]
     [-1.09329239e+00  1.91244750e-02  1.35093126e-02 -1.23403974e-03]
     [-1.06189264e+00  4.24409809e-02  9.80719337e-03 -1.93889113e-03]
     [-1.01158335e+00  5.62386943e-02  3.99051998e-03 -2.19201726e-03]
     [-9.53546157e-01  5.76436825e-02 -2.58553180e-03 -1.77986332e-03]
     [-9.00267870e-01  4.71330289e-02 -7.92512176e-03 -1.36770939e-03]]
    5.912839775437061e-05
    5.912839775437061e-05


More information can be found in the code documentation, which can be accessed, for example, via

`?CubicSpline`

`?CubicSpline.coeff`

## Bicubic Spline (2D)

The class `BicubicSpline` provides a method for interpolating data sampled on a uniform two-dimensional grid.


```python
from multispline.spline import BicubicSpline
import numpy as np
import matplotlib.pyplot as plt
```

As an example, we demonstrate how we can use this class to interpolate a two-dimensional test function


```python
def test_function_2d(x, y):
    return np.sin(x) * np.cos(4*y)

NX = 100
NY = 101
sample_points_x = np.linspace(0, 5, NX)
sample_points_y = np.linspace(0, 5, NY)
sample_grid_xy = np.meshgrid(sample_points_x, sample_points_y, indexing='ij')
sample_values_z = test_function_2d(*sample_grid_xy)
```

Once again, interpolating the data is as simple as instantiating `BicubicSpline`. Note that the grid points are passed in as 1D-arrays, while the sample values must be passed on a 2D grid with dimensions (NX, NY)


```python
bspl = BicubicSpline(sample_points_x, sample_points_y, sample_values_z)
```

We can visually inspect the performance of our interpolating function in one dimension by fixing the value of $x$ or $y$


```python
plt.plot(sample_points_y, sample_values_z[10], '.', label='original')
plt.plot(sample_points_y, bspl(sample_points_x[10], sample_points_y), label='spline')
plt.legend()
plt.xlabel('y')
plt.ylabel(f'f({str(sample_points_x[10])[:7]}, y)')
plt.show()
```


    
![png](tutorial_files/tutorial_36_0.png)
    


Alternatively, we can compare the spline against the test function on a grid that differs from the sample grid


```python
test_points_x = np.linspace(0, 5, 66)
test_points_y = np.linspace(0, 5, 66)
test_grid_xy = np.meshgrid(test_points_x, test_points_y, indexing='ij')
test_values_z = test_function_2d(*test_grid_xy)
bspl_values_z = bspl(*test_grid_xy)
```

Plotting the absolute error in two-dimensions, we see good agreement between the spline and the exact solution


```python
from matplotlib.colors import LogNorm
plt.pcolormesh(*test_grid_xy, np.abs(bspl_values_z - test_values_z), shading='gouraud', norm=LogNorm())
cbar = plt.colorbar()
cbar.set_label('Absolute error', rotation=270, labelpad=15)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```


    
![png](tutorial_files/tutorial_40_0.png)
    


Like `CubicSpline`, a `BicubicSpline` can be created using different boundary conditions. At the moment, the code requires that the same boundary condition is used for all dimensions.

One can also evaluate up to two partial derivatives of the spline via the class methods `deriv_x`, `deriv_y`, `deriv_xx`, `deriv_yy`, and `deriv_xy`.

Lastly, one can access spline coefficients through the class property `coefficients` or method `coeff`

## Tricubic Spline (3D)

The class `TricubicSpline` provides a method for interpolating data sampled on a uniform three-dimensional grid.


```python
from multispline.spline import TricubicSpline
```

As before, we demonstrate its use by interpolating a three-dimensional test function


```python
def test_function_3d(x, y, z):
    return np.sin(x) * np.cos(4*y) + np.i0(3.2*z)

NX = 65
NY = 100
NZ = 85
sample_points_x = np.linspace(0, 5, NX)
sample_points_y = np.linspace(0, 5, NY)
sample_points_z = np.linspace(-2, 2, NZ)
sample_points_grid = np.meshgrid(sample_points_x, sample_points_y, sample_points_z, indexing='ij')
sample_values_f = test_function_3d(*sample_points_grid)
```

Like the other spline classes, the spline is created by instantiating `TricubicSpline`. The grid points are passed in as 1D-arrays, while the sample values must be passed on a 3D grid with dimensions (NX, NY, NZ)


```python
tspl = TricubicSpline(sample_points_x, sample_points_y, sample_points_z, sample_values_f)
```

We can compare the spline against the test function on a grid that differs from the sample grid


```python
test_points_x = np.linspace(0, 5, 77)
test_points_y = np.linspace(0, 5, 81)
test_points_z = np.linspace(-2, 2, 59)
test_points_grid = np.meshgrid(test_points_x, test_points_y, test_points_z, indexing='ij')
test_values_f = test_function_3d(*test_points_grid)
tspl_values_f = tspl(*test_points_grid)
```

To make it easier to visualize the comparison, we take the maximum error along the x-axis to reduce the data down to two dimensions.


```python
average_error = np.max(np.abs(tspl_values_f - test_values_f), axis=0)
plot_grid = np.meshgrid(test_points_y, test_points_z, indexing='ij')
```

Making it easy to plot the results


```python
from matplotlib.colors import LogNorm
plt.pcolormesh(*plot_grid, average_error, shading='gouraud', norm=LogNorm())
cbar = plt.colorbar()
cbar.set_label('Max absolute error in x', rotation=270, labelpad=15)
plt.xlabel('y')
plt.ylabel('z')
plt.show()
```


    
![png](tutorial_files/tutorial_54_0.png)
    


Like the other classes, a `TricubicSpline` can be created using different boundary conditions. At the moment, the code requires that the same boundary condition is used for all dimensions.

One can also evaluate up to two partial derivatives of the spline via the class methods `deriv_x`, `deriv_y` , `deriv_z`, `deriv_xx`, `deriv_yy`, `deriv_zz`, `deriv_xy`, `deriv_yz`, and `deriv_xz`.

Lastly, one can access spline coefficients through the class property `coefficients` or method `coeff`
