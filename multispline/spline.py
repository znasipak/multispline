from splinecy import CyCubicSpline, CyBicubicSpline, CyTricubicSpline, CyQuadcubicSpline
import numpy as np

cubic_spline_bc_dict = {
    "natural": 0,
    "not-a-knot": 1,
    "clamped": 2,
    "E(3)": 3,
    # "natural-alt": 4 
}

def available_boundary_conditions():
    """
    Returns the available boundary conditions for cubic splines
    
    Returns
    -------
    list[str]
    """
    return list(cubic_spline_bc_dict.keys())
    
class CubicSplineUniformGrid:
    """
    A class for producing a cubic spline of a function :math:`f(x)` given its values
    :math:`f_{i} = f(x_i)` where :math:`x_i = x_0 + i*dx` is a grid of :math:`(N+1)` uniformly-spaced
    points with spacing :math:`dx`.
    
    Parameters
    ----------
    x : double
        The starting point of the grid
    dx : double
        The spacing between grid points
    f : 1d-array[double]
        Function values corresponding to the grid points :math:`x_i`
    bc : str (optional)
        Boundary value method. Valid options include "natural", "not-a-knot", "clamped", and "E(3)"
    """
    def __init__(self, x0, dx, f, bc = "E(3)"):
        self.boundary_conditions_dict = cubic_spline_bc_dict
        self.available_boundary_conditions = self.boundary_conditions_dict.keys()

        assert isinstance(f, np.ndarray)

        self.x0 = x0
        self.dx = dx
        self.nx = f.shape[0] - 1

        self.check_boundary_conditions(bc)
        
        self.base = CyCubicSpline(self.x0, self.dx, np.ascontiguousarray(f), self.boundary_conditions_dict[bc])
    
    def check_boundary_conditions(self, method):
        if method not in self.available_boundary_conditions:
            raise ValueError("No available method " + method)

    @property
    def coefficients(self):
        """
        The 2D array of spline coefficients with dimensions :code:`(nx, 4)`.
        Data are ordered so that the element at index :code:`(i, mx)`
        returns the same value as :code:`coeffs(i, mx)`

        Returns
        -------
        3d-array[double]
        """
        return np.array([[self.base.coefficient(i, j) for j in range(4)] for i in range(self.nx)])

    def coeff(self, i, mx):
        """
        Returns the spline coefficients :math:`c_{i}^{(m_x)}` defined by the
        spline :math:`f_{i}`

        .. math::
            f_{i}(x) = \\sum_{m_x=0}^3 c_{i}^{(m_x)}(x-x_i)^{m_x}

        Parameters
        ----------
        i : int
            Coefficient for the domain :math:`x_i \\leq x \\leq x_{i+1}`
        mx : int
            Coefficient weighting :math:`(x-x_i)^{m_x}` in the spline series
    
        Returns
        -------
        double
        """
        return self.base.coefficient(i, mx)

    def eval(self, x):
        """
        Evaluates the spline at the point x

        Parameters
        ----------
        x : double or 1d-array[double]
            dependent parameter

        Returns
        -------
        double or 1d-array[double]
        """
        if isinstance(x, np.ndarray):
            return np.array([self.base.eval(xi) for xi in x])
        return self.base.eval(x)
    
    def deriv(self, x):
        """
        Evaluates the derivative of the spline at the point x

        Parameters
        ----------
        x : double or 1d-array[double]
            dependent parameter

        Returns
        -------
        double or 1d-array[double]
        """
        if isinstance(x, np.ndarray):
            return np.array([self.base.deriv(xi) for xi in x])
        return self.base.deriv(x)
    
    def deriv2(self, x):
        """
        Evaluates the second derivative of the spline at the point x

        Parameters
        ----------
        x : double or 1d-array[double]
            dependent parameter

        Returns
        -------
        double or 1d-array[double]
        """
        if isinstance(x, np.ndarray):
            return np.array([self.base.deriv2(xi) for xi in x])
        return self.base.deriv2(x)

    def to_numba(self):
        """
        Returns a numba ``jitclass`` view of this spline that can be constructed
        and evaluated from within ``@njit``-compiled functions.

        The returned object exposes ``eval``, ``deriv`` and ``deriv2`` methods
        matching this class. The spline coefficients are computed once here (by
        the C++ backend); only the evaluation runs under numba.

        Returns
        -------
        multispline.numba.CubicSplineNumba

        Notes
        -----
        Requires the optional ``numba`` dependency
        (``pip install multispline[numba]``).
        """
        from .numba import CubicSplineNumba
        c = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        return CubicSplineNumba(c, float(self.x0), float(self.dx), int(self.nx))

    def to_jax(self):
        """
        Returns a JAX pytree view of this spline that can be evaluated and
        differentiated inside ``jax.jit``/``jax.vmap``/``jax.grad``.

        The returned object exposes ``eval``, ``deriv`` and ``deriv2`` methods;
        derivatives are computed by automatic differentiation of the value kernel.
        The spline coefficients are computed once here (by the C++ backend); only
        the evaluation runs under JAX.

        Returns
        -------
        multispline.jax.CubicSplineJax

        Notes
        -----
        Requires the optional ``jax`` dependency (``pip install multispline[jax]``)
        and 64-bit mode (``jax.config.update("jax_enable_x64", True)``) for results
        that match the C++ backend to double precision.
        """
        from . import jax as _msjax
        _msjax._warn_if_no_x64()
        c = _msjax.jnp.asarray(self.coefficients)
        return _msjax.CubicSplineJax(c, _msjax.jnp.asarray(self.x0), _msjax.jnp.asarray(self.dx))

    def __call__(self, x):
        """
        Evaluates the spline at the point x

        Parameters
        ----------
        x : double or 1d-array[double]
            dependent parameter

        Returns
        -------
        double or 1d-array[double]
        """
        return self.eval(x)

class CubicSpline(CubicSplineUniformGrid):
    """
    A class for producing a cubic spline of a function :math:`f(x)` given its values
    :math:`f_{i} = f(x_i)` where :math:`x_i = x_0, x_1, \\dots , x_N` is a grid of :math:`(N+1)` uniformly-spaced
    points.
    
    Parameters
    ----------
    x : 1d-array[double]
        A uniformly-spaced grid of points
    f : 1d-array[double]
        Function values corresponding to the grid points x
    bc : str (optional)
        Boundary value method. Valid options include "natural", "not-a-knot", "clamped", and "E(3)"
    """
    def __init__(self, x, f, bc = "E(3)"):
        self.boundary_conditions_dict = cubic_spline_bc_dict
        self.available_boundary_conditions = self.boundary_conditions_dict.keys()

        assert isinstance(x, np.ndarray)
        assert isinstance(f, np.ndarray)
        assert x.shape == f.shape, "Shapes of arrays {} and {} do not match".format(x.shape, f.shape)

        self.x0 = x[0]
        self.dx = x[1] - self.x0
        self.nx = f.shape[0] - 1

        dx_array = x[1:] - x[:-1]
        assert np.allclose(dx_array, self.dx*np.ones(dx_array.shape[0])), "Sampling points are not evenly spaced"
        self.check_boundary_conditions(bc)
        
        self.base = CyCubicSpline(self.x0, self.dx, np.ascontiguousarray(f), self.boundary_conditions_dict[bc])
    
# Bicubic spline
class BicubicSpline:
    """
    A class for producing a bicubic spline of a function :math:`f(x, y)` given its values
    :math:`f_{ij} = f(x_i, y_j)` where :math:`x_i = x_0, x_1, \\dots , x_N` is a grid of :math:`(N+1)` uniformly-spaced
    points and :math:`y_j = y_0, y_1, \\dots , y_M` is a grid of :math:`(M+1)` uniformly-spaced
    points. The input :math:`f_{ij}` is therefore structured as a :math:`(N+1) \\times (M+1)` matrix of function values 
    
    .. math::
        \\begin{align*}
        f(x_i, y_j) &= 
            \\begin{pmatrix}
                f_{00} & f_{01} & \\cdots & f_{0M}
                \\\\
                f_{10} & f_{11} & \\cdots & f_{1M}
                \\\\
                \\vdots  &  \\vdots & \\ddots & \\vdots
                \\\\
                f_{N0} & f_{N1} & \\cdots & f_{NM}
            \\end{pmatrix}
        \\end{align*}
    
    
    Parameters
    ----------
    x : 1d-array[double]
        A uniformly-spaced grid of points
    y : 1d-array[double]
        A uniformly-spaced grid of points
    f : 2d-array[double]
        Function values corresponding to the grid points x, y
    bc : str (optional)
        Boundary value method. Valid options include "natural", "not-a-knot", "clamped", and "E(3)"
    """
    def __init__(self, x, y, f, bc = "E(3)"):
        self.boundary_conditions_dict = cubic_spline_bc_dict
        self.available_boundary_conditions = self.boundary_conditions_dict.keys()
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(f, np.ndarray)
        assert (x.shape[0], y.shape[0]) == (f.shape[0], f.shape[1]), "Shapes of arrays {}, {}, and {} do not match".format(x.shape, y.shape, f.shape)

        self.x0 = x[0]
        self.y0 = y[0]
        self.dx = x[1]-self.x0
        self.dy = y[1]-self.y0
        self.nx = f.shape[0] - 1
        self.ny = f.shape[1] - 1

        dx_array = x[1:] - x[:-1]
        dy_array = y[1:] - y[:-1]
        assert np.allclose(dx_array, self.dx*np.ones(dx_array.shape[0])), "Sampling points in x are not evenly spaced"
        assert np.allclose(dy_array, self.dy*np.ones(dy_array.shape[0])), "Sampling points in y are not evenly spaced"

        self.check_boundary_conditions(bc)
        self.base = CyBicubicSpline(self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, np.ascontiguousarray(f), self.boundary_conditions_dict[bc])

    def check_boundary_conditions(self, method):
        if method not in self.available_boundary_conditions:
            raise ValueError("No available method " + method)
        
    @property
    def coefficients(self):
        """
        The 4D array of spline coefficients with dimensions :code:`(nx, ny, 4, 4)`.
        Data are ordered so that the element at index :code:`(i, j, mx, my)`
        returns the same value as :code:`coeffs(i, j, mx, my)`

        Returns
        -------
        3d-array[double]
        """
        return np.array([[[[self.base.coefficient(i, j, mx, my) for my in range(4)] for mx in range(4)] for j in range(self.ny)] for i in range(self.nx)])

    def eval(self, x, y):
        """
        Evaluates the spline at the point (x, y)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter

        Returns
        -------
        double
        """
        ## allow for numpy broadcasting
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            b = np.broadcast(x, y)
            out = np.empty(b.shape)
            out.flat = [self.base.eval(xi, yi) for (xi, yi) in b]
            return out
        return self.base.eval(x, y)

    def deriv_x(self, x, y):
        """
        Evaluates the partial derivative of the spline with respect to x at the point (x, y)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            b = np.broadcast(x, y)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_x(xi, yi) for (xi, yi) in b]
            return out
        return self.base.deriv_x(x, y)
    
    def deriv_y(self, x, y):
        """
        Evaluates the partial derivative of the spline with respect to y at the point (x, y)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            b = np.broadcast(x, y)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_y(xi, yi) for (xi, yi) in b]
            return out
        return self.base.deriv_y(x, y)
    
    def deriv_xx(self, x, y):
        """
        Evaluates the second partial derivative of the spline with respect to x at the point (x, y)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            b = np.broadcast(x, y)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_xx(xi, yi) for (xi, yi) in b]
            return out
        return self.base.deriv_xx(x, y)
    
    def deriv_yy(self, x, y):
        """
        Evaluates the second partial derivative of the spline with respect to y at the point (x, y)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            b = np.broadcast(x, y)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_yy(xi, yi) for (xi, yi) in b]
            return out
        return self.base.deriv_yy(x, y)
    
    def deriv_xy(self, x, y):
        """
        Evaluates the mixed partial derivative of the spline with respect to x and y at the point (x, y)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            b = np.broadcast(x, y)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_xy(xi, yi) for (xi, yi) in b]
            return out
        return self.base.deriv_xy(x, y)
    
    def coeff(self, i, j, mx, my):
        """
        Returns the spline coefficients :math:`c_{ij}^{(m_x,m_y)}` defined by the
        spline :math:`f_{ij}`

        .. math::
            f_{ij}(x, y) = \\sum_{m_x=0}^3 \\sum_{m_y=0}^3 c_{ij}^{(m_x,m_y)}(x-x_i)^{m_x}(y-y_j)^{m_y}

        Parameters
        ----------
        i : int
            Coefficient for the domain :math:`x_i \\leq x \\leq x_{i+1}`
        j : int
            Coefficient for the domain :math:`y_j \\leq y \\leq y_{j+1}`
        mx : int
            Coefficient weighting :math:`(x-x_i)^{m_x}` in the spline series
        my : int
            Coefficient weighting :math:`(y-y_j)^{m_y}` in the spline series
    
        Returns
        -------
        double
        """
        return self.base.coefficient(i, j, mx, my)

    def to_numba(self):
        """
        Returns a numba ``jitclass`` view of this spline that can be constructed
        and evaluated from within ``@njit``-compiled functions.

        The returned object exposes ``eval`` and the ``deriv_*`` methods matching
        this class. The spline coefficients are computed once here (by the C++
        backend); only the evaluation runs under numba.

        Returns
        -------
        multispline.numba.BicubicSplineNumba

        Notes
        -----
        Requires the optional ``numba`` dependency
        (``pip install multispline[numba]``).
        """
        from .numba import BicubicSplineNumba
        c = np.ascontiguousarray(self.coefficients, dtype=np.float64)
        return BicubicSplineNumba(c, float(self.x0), float(self.dx), int(self.nx),
                                  float(self.y0), float(self.dy), int(self.ny))

    def to_jax(self):
        """
        Returns a JAX pytree view of this spline that can be evaluated and
        differentiated inside ``jax.jit``/``jax.vmap``/``jax.grad``.

        The returned object exposes ``eval`` and the ``deriv_*`` methods;
        derivatives are computed by automatic differentiation of the value kernel.
        The spline coefficients are computed once here (by the C++ backend); only
        the evaluation runs under JAX.

        Returns
        -------
        multispline.jax.BicubicSplineJax

        Notes
        -----
        Requires the optional ``jax`` dependency (``pip install multispline[jax]``)
        and 64-bit mode (``jax.config.update("jax_enable_x64", True)``) for results
        that match the C++ backend to double precision.
        """
        from . import jax as _msjax
        _msjax._warn_if_no_x64()
        c = _msjax.jnp.asarray(self.coefficients)
        return _msjax.BicubicSplineJax(
            c, _msjax.jnp.asarray(self.x0), _msjax.jnp.asarray(self.dx),
            _msjax.jnp.asarray(self.y0), _msjax.jnp.asarray(self.dy))

    def __call__(self, x, y):
        return self.eval(x, y)

class TricubicSpline:
    """
    A class for producing a tricubic spline of a function :math:`f(x, y, z)` given its values
    :math:`f_{ijk} = f(x_i, y_j, z_k)` where :math:`x_i = x_0, x_1, \\dots , x_N` is a grid of :math:`(N+1)` uniformly-spaced
    points, :math:`y_j = y_0, y_1, \\dots , y_M` is a grid of :math:`(M+1)` uniformly-spaced
    points, and :math:`z_k = z_0, z_1, \\dots , z_O` is a grid of :math:`(O+1)` uniformly-spaced
    points. The input :math:`f_{ijk}` is therefore structured as a :math:`(N+1) \\times (M+1) \\times (O+1)` tensor of function values 
    
    .. math::
        \\begin{align*}
        f(x_i, y_j,z_k) &= 
            \\begin{pmatrix}
                f_{00k} & f_{01k} & \\cdots & f_{0Mk}
                \\\\
                f_{10k} & f_{11k} & \\cdots & f_{1Mk}
                \\\\
                \\vdots  &  \\vdots & \\ddots & \\vdots
                \\\\
                f_{N0k} & f_{N1k} & \\cdots & f_{NMk}
            \\end{pmatrix}
        \\end{align*}

    where each entry is a vector of length :math:`O+1` in the :math:`z`-dimension
    
    Parameters
    ----------
    x : 1d-array[double]
        A uniformly-spaced grid of points
    y : 1d-array[double]
        A uniformly-spaced grid of points
    z : 1d-array[double]
        A uniformly-spaced grid of points
    f : 3d-array[double]
        Function values corresponding to the grid points x, y, z or pre-computed spline coefficients
    bc : str (optional)
        Boundary value method. Valid options include "natural", "not-a-knot", "clamped", and "E(3)"
    """
    def __init__(self, x, y, z, f, bc = "E(3)"):
        self.boundary_conditions_dict = cubic_spline_bc_dict
        self.available_boundary_conditions = self.boundary_conditions_dict.keys()
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(z, np.ndarray)
        assert isinstance(f, np.ndarray)
        assert ((x.shape[0], y.shape[0], z.shape[0]) == (f.shape[0], f.shape[1], f.shape[2]) or (x.shape[0] - 1, y.shape[0] - 1, 64*(z.shape[0] - 1)) == (f.shape[0], f.shape[1], f.shape[2])), "Shapes of arrays {}, {}, {}, and {} do not match".format(x.shape, y.shape, z.shape, f.shape)

        self.x0 = x[0]
        self.y0 = y[0]
        self.z0 = z[0]
        self.dx = x[1]-self.x0
        self.dy = y[1]-self.y0
        self.dz = z[1]-self.z0
        self.nx = x.shape[0] - 1
        self.ny = y.shape[0] - 1
        self.nz = z.shape[0] - 1

        dx_array = np.diff(x)
        dy_array = np.diff(y)
        dz_array = np.diff(z)

        assert np.allclose(dx_array, self.dx*np.ones(dx_array.shape[0])), "Sampling points in x are not evenly spaced"
        assert np.allclose(dy_array, self.dy*np.ones(dy_array.shape[0])), "Sampling points in y are not evenly spaced"
        assert np.allclose(dz_array, self.dz*np.ones(dz_array.shape[0])), "Sampling points in z are not evenly spaced"

        self.check_boundary_conditions(bc)
        self.base = CyTricubicSpline(self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, np.ascontiguousarray(f), self.boundary_conditions_dict[bc])

    def check_boundary_conditions(self, method):
        if method not in self.available_boundary_conditions:
            raise ValueError("No available method " + method)

    def eval(self, x, y, z):
        """
        Evaluates the spline at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.eval(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.eval(x, y, z)

    def deriv_x(self, x, y, z):
        """
        Evaluates the partial derivative of the spline with respect to x at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_x(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_x(x, y, z)
    
    def deriv_y(self, x, y, z):
        """
        Evaluates the partial derivative of the spline with respect to y at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_y(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_y(x, y, z)
    
    def deriv_z(self, x, y, z):
        """
        Evaluates the partial derivative of the spline with respect to z at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_z(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_z(x, y, z)
    
    def deriv_xx(self, x, y, z):
        """
        Evaluates the second partial derivative of the spline with respect to x at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_xx(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_xx(x, y, z)
    
    def deriv_yy(self, x, y, z):
        """
        Evaluates the second partial derivative of the spline with respect to y at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_yy(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_yy(x, y, z)
    
    def deriv_zz(self, x, y, z):
        """
        Evaluates the second partial derivative of the spline with respect to z at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_zz(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_zz(x, y, z)
    
    def deriv_xy(self, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to x and y at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_xy(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_xy(x, y, z)
    
    def deriv_xz(self, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to x and z at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_xz(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_xz(x, y, z)
    
    def deriv_yz(self, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to y and z at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_yz(xi, yi, zi) for (xi, yi, zi) in b]
            return out
        return self.base.deriv_yz(x, y, z)
    
    def coeff(self, i, j, k, mx, my, mz):
        """
        Returns the spline coefficients :math:`c_{ijk}^{(m_x,m_y,m_z)}` defined by the
        spline :math:`f_{ijk}:math:`

        .. math::
            f_{ijk}(x, y, z) = \\sum_{m_x=0}^3 \\sum_{m_y=0}^3 \\sum_{m_z=0}^3 c_{ijk}^{(m_x,m_y,m_z)}(x-x_i)^{m_x}(y-y_j)^{m_y}(z-z_k)^{m_z}

        Parameters
        ----------
        i : int
            Coefficient for the domain :math:`x_i \\leq x \\leq x_{i+1}`
        j : int
            Coefficient for the domain :math:`y_j \\leq y \\leq y_{j+1}`
        k : int
            Coefficient for the domain :math:`z_k \\leq z \\leq z_{k+1}`
        mx : int
            Coefficient weighting :math:`(x-x_i)^{m_x}` in the spline series
        my : int
            Coefficient weighting :math:`(y-y_j)^{m_y}` in the spline series
        mz : int
            Coefficient weighting :math:`(z-z_k)^{m_z}` in the spline series
    
        Returns
        -------
        double
        """
        return self.base.coefficient(i, j, k, mx, my, mz)
    
    @property
    def coefficients(self):
        """
        The 3D array of spline coefficients with dimensions :code:`(nx, ny, 64*nz)`.
        Data are ordered so that the element at index :code:`(i, j, k, 4*(4*(4*k + mx) + my) + mz)`
        returns the same value as :code:`coeffs(i, j, k, mx, my, mz)`

        Returns
        -------
        3d-array[double]
        """
        return self.base.coefficients()

    def to_numba(self):
        """
        Returns a numba ``jitclass`` view of this spline that can be constructed
        and evaluated from within ``@njit``-compiled functions.

        The returned object exposes ``eval`` and the ``deriv_*`` methods matching
        this class. The spline coefficients are computed once here (by the C++
        backend); only the evaluation runs under numba.

        Returns
        -------
        multispline.numba.TricubicSplineNumba

        Notes
        -----
        Requires the optional ``numba`` dependency
        (``pip install multispline[numba]``).
        """
        from .numba import TricubicSplineNumba
        c = np.ascontiguousarray(self.coefficients, dtype=np.float64).reshape(
            self.nx, self.ny, self.nz, 4, 4, 4)
        return TricubicSplineNumba(c, float(self.x0), float(self.dx), int(self.nx),
                                   float(self.y0), float(self.dy), int(self.ny),
                                   float(self.z0), float(self.dz), int(self.nz))

    def to_jax(self):
        """
        Returns a JAX pytree view of this spline that can be evaluated and
        differentiated inside ``jax.jit``/``jax.vmap``/``jax.grad``.

        The returned object exposes ``eval`` and the ``deriv_*`` methods;
        derivatives are computed by automatic differentiation of the value kernel.
        The spline coefficients are computed once here (by the C++ backend); only
        the evaluation runs under JAX.

        Returns
        -------
        multispline.jax.TricubicSplineJax

        Notes
        -----
        Requires the optional ``jax`` dependency (``pip install multispline[jax]``)
        and 64-bit mode (``jax.config.update("jax_enable_x64", True)``) for results
        that match the C++ backend to double precision.
        """
        from . import jax as _msjax
        _msjax._warn_if_no_x64()
        c = _msjax.jnp.asarray(self.coefficients).reshape(
            self.nx, self.ny, self.nz, 4, 4, 4)
        return _msjax.TricubicSplineJax(
            c, _msjax.jnp.asarray(self.x0), _msjax.jnp.asarray(self.dx),
            _msjax.jnp.asarray(self.y0), _msjax.jnp.asarray(self.dy),
            _msjax.jnp.asarray(self.z0), _msjax.jnp.asarray(self.dz))

    def __call__(self, x, y, z):
        """
        Evaluates the spline at the point (x, y, z)

        Parameters
        ----------
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        return self.eval(x, y, z)


class QuadcubicSpline:
    """
    A class for producing a quadcubic spline of a function :math:`f(w, x, y, z)` given its values
    :math:`f_{hijk} = f(w_h, x_i, y_j, z_k)` where :math:`w_h = w_0, w_1, \\dots , w_L` is a grid of :math:`(L+1)` uniformly-spaced
    points, :math:`x_i = x_0, x_1, \\dots , x_M` is a grid of :math:`(M+1)` uniformly-spaced
    points, :math:`y_j = y_0, y_1, \\dots , y_N` is a grid of :math:`(N+1)` uniformly-spaced
    points, and :math:`z_k = z_0, z_1, \\dots , z_O` is a grid of :math:`(O+1)` uniformly-spaced
    points. The input :math:`f_{hijk}` is therefore structured as a :math:`(L+1) \\times (M+1) \\times (N+1) \\times (O+1)` tensor of function values
    
    Parameters
    ----------
    w : 1d-array[double]
        A uniformly-spaced grid of points
    x : 1d-array[double]
        A uniformly-spaced grid of points
    y : 1d-array[double]
        A uniformly-spaced grid of points
    z : 1d-array[double]
        A uniformly-spaced grid of points
    f : 4d-array[double]
        Function values corresponding to the grid points w, x, y, z or pre-computed spline coefficients
    bc : str (optional)
        Boundary value method. Valid options include "natural", "not-a-knot", "clamped", and "E(3)"
    """
    def __init__(self, w, x, y, z, f, bc = "E(3)"):
        self.boundary_conditions_dict = cubic_spline_bc_dict
        self.available_boundary_conditions = self.boundary_conditions_dict.keys()
        assert isinstance(w, np.ndarray)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(z, np.ndarray)
        assert isinstance(f, np.ndarray)
        assert ((w.shape[0], x.shape[0], y.shape[0], z.shape[0]) == (f.shape[0], f.shape[1], f.shape[2], f.shape[3]) or (w.shape[0] - 1, x.shape[0] - 1, y.shape[0] - 1, 256*(z.shape[0] - 1)) == (f.shape[0], f.shape[1], f.shape[2], f.shape[3])), "Shapes of arrays {}, {}, {}, {}, and {} do not match".format(w.shape, x.shape, y.shape, z.shape, f.shape)

        self.w0 = w[0]
        self.x0 = x[0]
        self.y0 = y[0]
        self.z0 = z[0]

        self.dw = w[1]-self.w0
        self.dx = x[1]-self.x0
        self.dy = y[1]-self.y0
        self.dz = z[1]-self.z0

        self.nw = w.shape[0] - 1
        self.nx = x.shape[0] - 1
        self.ny = y.shape[0] - 1
        self.nz = z.shape[0] - 1

        dw_array = np.diff(w)
        dx_array = np.diff(x)
        dy_array = np.diff(y)
        dz_array = np.diff(z)

        assert np.allclose(dw_array, self.dw*np.ones(dw_array.shape[0])), "Sampling points in w are not evenly spaced"
        assert np.allclose(dx_array, self.dx*np.ones(dx_array.shape[0])), "Sampling points in x are not evenly spaced"
        assert np.allclose(dy_array, self.dy*np.ones(dy_array.shape[0])), "Sampling points in y are not evenly spaced"
        assert np.allclose(dz_array, self.dz*np.ones(dz_array.shape[0])), "Sampling points in z are not evenly spaced"

        self.check_boundary_conditions(bc)
        self.base = CyQuadcubicSpline(self.w0, self.dw, self.nw, self.x0, self.dx, self.nx, self.y0, self.dy, self.ny, self.z0, self.dz, self.nz, np.ascontiguousarray(f), self.boundary_conditions_dict[bc])

    def check_boundary_conditions(self, method):
        if method not in self.available_boundary_conditions:
            raise ValueError("No available method " + method)

    def eval(self, w, x, y, z):
        """
        Evaluates the spline at the point (w, x, y, z)

        Parameters
        ----------
        w: double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.eval(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.eval(w, x, y, z)

    def deriv_w(self, w, x, y, z):
        """
        Evaluates the partial derivative of the spline with respect to w at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_w(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_w(w, x, y, z)

    def deriv_x(self, w, x, y, z):
        """
        Evaluates the partial derivative of the spline with respect to x at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_x(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_x(w, x, y, z)

    def deriv_y(self, w, x, y, z):
        """
        Evaluates the partial derivative of the spline with respect to y at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_y(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_y(w, x, y, z)

    def deriv_z(self, w, x, y, z):
        """
        Evaluates the partial derivative of the spline with respect to z at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_z(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_z(w, x, y, z)

    def deriv_ww(self, w, x, y, z):
        """
        Evaluates the second partial derivative of the spline with respect to w at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_ww(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_ww(w, x, y, z)

    def deriv_xx(self, w, x, y, z):
        """
        Evaluates the second partial derivative of the spline with respect to x at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_xx(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_xx(w, x, y, z)
    
    def deriv_yy(self, w, x, y, z):
        """
        Evaluates the second partial derivative of the spline with respect to y at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_yy(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_yy(w, x, y, z)

    def deriv_zz(self, w, x, y, z):
        """
        Evaluates the second partial derivative of the spline with respect to z at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_zz(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_zz(w, x, y, z)

    def deriv_wx(self, w, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to w and x at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_wx(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_wx(w, x, y, z)

    def deriv_wy(self, w, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to w and y at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_wy(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_wy(w, x, y, z)

    def deriv_wz(self, w, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to w and z at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_wz(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_wz(w, x, y, z)

    def deriv_xy(self, w, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to x and y at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_xy(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_xy(w, x, y, z)
    
    def deriv_xz(self, w, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to x and z at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_xz(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_xz(w, x, y, z)
    
    def deriv_yz(self, w, x, y, z):
        """
        Evaluates the mixed partial derivative of the spline with respect to y and z at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        if isinstance(w, np.ndarray) or isinstance(x, np.ndarray) or isinstance(y, np.ndarray) or isinstance(z, np.ndarray):
            b = np.broadcast(w, x, y, z)
            out = np.empty(b.shape)
            out.flat = [self.base.deriv_yz(wi, xi, yi, zi) for (wi, xi, yi, zi) in b]
            return out
        return self.base.deriv_yz(w, x, y, z)

    def coeff(self, h, i, j, k, mw, mx, my, mz):
        """
        Returns the spline coefficients :math:`c_{hijk}^{(mw,m_x,m_y,m_z)}` defined by the
        spline :math:`f_{hijk}:math:`

        .. math::
            f_{hijk}(w, x, y, z) = \\sum_{m_w=0}^3\\sum_{m_x=0}^3 \\sum_{m_y=0}^3 \\sum_{m_z=0}^3 c_{hijk}^{(m_w,m_x,m_y,m_z)}(w-w_h)^{m_w}(x-x_i)^{m_x}(y-y_j)^{m_y}(z-z_k)^{m_z}

        Parameters
        ----------
        h : int
            Coefficient for the domain :math:`w_h \\leq w \\leq w_{h+1}`
        i : int
            Coefficient for the domain :math:`x_i \\leq x \\leq x_{i+1}`
        j : int
            Coefficient for the domain :math:`y_j \\leq y \\leq y_{j+1}`
        k : int
            Coefficient for the domain :math:`z_k \\leq z \\leq z_{k+1}`
        mw : int
            Coefficient weighting :math:`(w-w_h)^{m_w}` in the spline series
        mx : int
            Coefficient weighting :math:`(x-x_i)^{m_x}` in the spline series
        my : int
            Coefficient weighting :math:`(y-y_j)^{m_y}` in the spline series
        mz : int
            Coefficient weighting :math:`(z-z_k)^{m_z}` in the spline series
    
        Returns
        -------
        double
        """
        return self.base.coefficient(h, i, j, k, mw, mx, my, mz)
    
    @property
    def coefficients(self):
        """
        The 4D array of spline coefficients with dimensions :code:`(nw, nx, ny, 256*nz)`.
        Data are ordered so that the element at index :code:`(i, j, k, 4*(4*(4*(4*k + mw) + mx) + my) + mz)`
        returns the same value as :code:`coeffs(h, i, j, k, mw, mx, my, mz)`

        Returns
        -------
        4d-array[double]
        """
        return self.base.coefficients()

    def to_numba(self):
        """
        Returns a numba ``jitclass`` view of this spline that can be constructed
        and evaluated from within ``@njit``-compiled functions.

        The returned object exposes ``eval`` and the ``deriv_*`` methods matching
        this class. The spline coefficients are computed once here (by the C++
        backend); only the evaluation runs under numba.

        Returns
        -------
        multispline.numba.QuadcubicSplineNumba

        Notes
        -----
        Requires the optional ``numba`` dependency
        (``pip install multispline[numba]``).
        """
        from .numba import QuadcubicSplineNumba
        c = np.ascontiguousarray(self.coefficients, dtype=np.float64).reshape(
            self.nw, self.nx, self.ny, self.nz, 4, 4, 4, 4)
        return QuadcubicSplineNumba(c, float(self.w0), float(self.dw), int(self.nw),
                                    float(self.x0), float(self.dx), int(self.nx),
                                    float(self.y0), float(self.dy), int(self.ny),
                                    float(self.z0), float(self.dz), int(self.nz))

    def to_jax(self):
        """
        Returns a JAX pytree view of this spline that can be evaluated and
        differentiated inside ``jax.jit``/``jax.vmap``/``jax.grad``.

        The returned object exposes ``eval`` and the ``deriv_*`` methods (including
        the w-direction derivatives); derivatives are computed by automatic
        differentiation of the value kernel. The spline coefficients are computed
        once here (by the C++ backend); only the evaluation runs under JAX.

        Returns
        -------
        multispline.jax.QuadcubicSplineJax

        Notes
        -----
        Requires the optional ``jax`` dependency (``pip install multispline[jax]``)
        and 64-bit mode (``jax.config.update("jax_enable_x64", True)``) for results
        that match the C++ backend to double precision.
        """
        from . import jax as _msjax
        _msjax._warn_if_no_x64()
        c = _msjax.jnp.asarray(self.coefficients).reshape(
            self.nw, self.nx, self.ny, self.nz, 4, 4, 4, 4)
        return _msjax.QuadcubicSplineJax(
            c, _msjax.jnp.asarray(self.w0), _msjax.jnp.asarray(self.dw),
            _msjax.jnp.asarray(self.x0), _msjax.jnp.asarray(self.dx),
            _msjax.jnp.asarray(self.y0), _msjax.jnp.asarray(self.dy),
            _msjax.jnp.asarray(self.z0), _msjax.jnp.asarray(self.dz))

    def __call__(self, w, x, y, z):
        """
        Evaluates the spline at the point (w, x, y, z)

        Parameters
        ----------
        w : double
            dependent parameter
        x : double
            dependent parameter
        y : double
            dependent parameter
        z : double
            dependent parameter

        Returns
        -------
        double
        """
        return self.eval(w, x, y, z)
