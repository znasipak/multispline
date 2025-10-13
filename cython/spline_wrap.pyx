import numpy as np
cimport numpy as np
from cython.operator import dereference
import os
import warnings
from libcpp.vector cimport vector

cdef extern from "utils.hpp":
    cdef cppclass Matrix:
        Matrix()
        Matrix(size_t n)
        Matrix(size_t n, size_t m)
        Matrix(size_t n, size_t m, vector[double] A)
        Matrix(size_t n, size_t m, double val)

        void set_value(size_t i, size_t j, double val)
        double& operator()(size_t i, size_t j)

    cdef cppclass ThreeTensor:
        ThreeTensor()
        ThreeTensor(size_t nx)
        ThreeTensor(size_t nx, size_t ny, size_t nz)
        ThreeTensor(size_t nx, size_t ny, size_t nz, double *A)
        ThreeTensor(size_t nx, size_t ny, size_t nz, vector[double] A)
        ThreeTensor(size_t nx, size_t ny, size_t nz, double val)

        size_t rows() const
        size_t cols() const
        size_t slcs() const
        size_t size() const

        void row_replace(size_t i, Matrix row)
        void col_replace(size_t j, Matrix col)
        void slc_replace(size_t k, Matrix slc)

        Matrix row(size_t i)
        vector[double] rowcol(size_t i, size_t j)
        vector[double] rowslc(size_t i, size_t k)
        Matrix col(size_t j)
        vector[double] colslc(size_t j, size_t k)
        Matrix slc(size_t k)

        void reshape(size_t nx, size_t ny, size_t nz)
        ThreeTensor reshaped(size_t nx, size_t ny, size_t nz) const

        void set_value(size_t i, size_t j, size_t k, double val)

        double& operator()(size_t i, size_t j, size_t k)

        vector[double] data()

    cdef cppclass FourTensor:
        FourTensor()
        FourTensor(size_t nw, size_t nx, size_t ny, size_t nz)
        FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, vector[double] A)
        FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, double *A)
        FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, double val)

        size_t dim(size_t i) const
        size_t size() const

        void reshape(size_t nw, size_t nx, size_t ny, size_t nz)
        FourTensor reshaped(size_t nw, size_t nx, size_t ny, size_t nz) const

        void set_value(size_t i, size_t j, size_t k, size_t l, double val)

        double& operator()(size_t i, size_t j, size_t k, size_t l)

        vector[double] data()

cdef extern from "cubic.hpp":
    cdef cppclass CubicSpline:
        CubicSpline(double x0, double dx, const vector[double] &y, int method) except +
        CubicSpline(const vector[double] &x, const vector[double] &y, int method) except +

        double getSplineCoefficient(int i, int j)

        double evaluate(const double x)
        double derivative(const double x)
        double derivative2(const double x)

cdef extern from "bicubic.hpp":
    cdef cppclass BicubicSpline:
        BicubicSpline(const vector[double] &x, const vector[double] &y, const Matrix &z, int method) except +
        BicubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, const Matrix &z, int method) except +
        
        double evaluate(const double x, const double y)
        double derivative_x(const double x, const double y)
        double derivative_y(const double x, const double y)
        double derivative_xy(const double x, const double y)
        double derivative_xx(const double x, const double y)
        double derivative_yy(const double x, const double y)
        CubicSpline reduce_x(const double x)
        CubicSpline reduce_y(const double y)
        double getSplineCoefficient(int i, int j, int nx, int ny)

cdef extern from "tricubic.hpp":
    cdef cppclass TricubicSpline:
        TricubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, ThreeTensor &f, int method) except +
        
        double evaluate(const double x, const double y, const double z)
        double derivative_x(const double x, const double y, const double z)
        double derivative_y(const double x, const double y, const double z)
        double derivative_z(const double x, const double y, const double z)
        double derivative_xy(const double x, const double y, const double z)
        double derivative_xz(const double x, const double y, const double z)
        double derivative_yz(const double x, const double y, const double z)
        double derivative_xx(const double x, const double y, const double z)
        double derivative_yy(const double x, const double y, const double z)
        double derivative_zz(const double x, const double y, const double z)

        double getSplineCoefficient(int i, int j, int k, int nx, int ny, int nz)
        ThreeTensor getSplineCoefficients()

cdef extern from "quadcubic.hpp":
    cdef cppclass QuadcubicSpline:
        QuadcubicSpline(double w0, double dw, int nw, double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, FourTensor &f, int method) except +
        
        double evaluate(const double w, const double x, const double y, const double z)
        double derivative_w(const double w, const double x, const double y, const double z)
        double derivative_x(const double w, const double x, const double y, const double z)
        double derivative_y(const double w, const double x, const double y, const double z)
        double derivative_z(const double w, const double x, const double y, const double z)
        double derivative_wx(const double w, const double x, const double y, const double z)
        double derivative_wy(const double w, const double x, const double y, const double z)
        double derivative_wz(const double w, const double x, const double y, const double z)
        double derivative_xy(const double w, const double x, const double y, const double z)
        double derivative_xz(const double w, const double x, const double y, const double z)
        double derivative_yz(const double w, const double x, const double y, const double z)
        double derivative_ww(const double w, const double x, const double y, const double z)
        double derivative_xx(const double w, const double x, const double y, const double z)
        double derivative_yy(const double w, const double x, const double y, const double z)
        double derivative_zz(const double w, const double x, const double y, const double z)

        double getSplineCoefficient(int h, int i, int j, int k, int nw, int nx, int ny, int nz)
        FourTensor getSplineCoefficients()

cdef class CyCubicSpline:
    cdef CubicSpline *scpp

    def __init__(self, double x0, double dx, np.ndarray[ndim=1, dtype=np.float64_t, mode='c'] f, int method):
        cdef vector[double] fvec = vector[double](len(f))
        for i in range(len(f)):
            fvec[i] = f[i]
        self.scpp = new CubicSpline(x0, dx, fvec, method)

    def coefficient(self, int i, int j):
        return self.scpp.getSplineCoefficient(i, j)

    def eval(self, double x):
        return self.scpp.evaluate(x)

    def deriv(self, double x):
        return self.scpp.derivative(x)

    def deriv2(self, double x):
        return self.scpp.derivative2(x)
    

cdef class CyBicubicSpline:
    cdef BicubicSpline *scpp

    def __init__(self, double x0, double dx, int nx, double y0, double dy, int ny, np.ndarray[ndim=2, dtype=np.float64_t, mode='c'] f, int method):
        cdef Matrix mz = Matrix(nx + 1, ny + 1)
        for i in range(nx + 1):
            for j in range(ny + 1):
                mz.set_value(i, j, f[i, j])
        self.scpp = new BicubicSpline(x0, dx, nx, y0, dy, ny, mz, method)

    def coefficient(self, int i, int j, int nx, int ny):
        return self.scpp.getSplineCoefficient(i, j, nx, ny)
    
    def eval(self, double x, double y):
        return self.scpp.evaluate(x, y)

    def deriv_x(self, double x, double y):
        return self.scpp.derivative_x(x, y)

    def deriv_y(self, double x, double y):
        return self.scpp.derivative_y(x, y)
    
    def deriv_xx(self, double x, double y):
        return self.scpp.derivative_xx(x, y)

    def deriv_yy(self, double x, double y):
        return self.scpp.derivative_yy(x, y)

    def deriv_xy(self, double x, double y):
        return self.scpp.derivative_xy(x, y)

cdef class CyTricubicSpline:
    cdef TricubicSpline *scpp
    cdef int nx
    cdef int ny
    cdef int nz

    def __init__(self, double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, np.ndarray[ndim=3, dtype=np.float64_t, mode='c'] f, int method):
        cdef int fnx, fny, fnz
        fnx = f.shape[0]
        fny = f.shape[1]
        fnz = f.shape[2]
        self.nx = nx
        self.ny = ny
        self.nz = nz
        cdef ThreeTensor ftens = ThreeTensor(fnx, fny, fnz, &f[0,0,0])
        self.scpp = new TricubicSpline(x0, dx, nx, y0, dy, ny, z0, dz, nz, ftens, method)

    def coefficient(self, int i, int j, int k, int nx, int ny, int nz):
        return self.scpp.getSplineCoefficient(i, j, k, nx, ny, nz)

    def coefficients(self):
        cdef np.ndarray[ndim=3, dtype=np.float64_t, mode='c'] cijk = np.asarray(self.scpp.getSplineCoefficients().data()).reshape(self.nx, self.ny, 64*self.nz)
        return cijk

    def eval(self, double x, double y, double z):
        return self.scpp.evaluate(x, y, z)

    def deriv_x(self, double x, double y, double z):
        return self.scpp.derivative_x(x, y, z)

    def deriv_y(self, double x, double y, double z):
        return self.scpp.derivative_y(x, y, z)

    def deriv_z(self, double x, double y, double z):
        return self.scpp.derivative_z(x, y, z)
    
    def deriv_xx(self, double x, double y, double z):
        return self.scpp.derivative_xx(x, y, z)

    def deriv_yy(self, double x, double y, double z):
        return self.scpp.derivative_yy(x, y, z)

    def deriv_zz(self, double x, double y, double z):
        return self.scpp.derivative_yy(x, y, z)

    def deriv_xy(self, double x, double y, double z):
        return self.scpp.derivative_xy(x, y, z)

    def deriv_xz(self, double x, double y, double z):
        return self.scpp.derivative_xz(x, y, z)

    def deriv_yz(self, double x, double y, double z):
        return self.scpp.derivative_yz(x, y, z)

cdef class CyQuadcubicSpline:
    cdef QuadcubicSpline *scpp
    cdef int nw
    cdef int nx
    cdef int ny
    cdef int nz

    def __init__(self, double w0, double dw, int nw, double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, np.ndarray[ndim=4, dtype=np.float64_t, mode='c'] f, int method):
        cdef int fnw, fnx, fny, fnz
        fnw = f.shape[0]
        fnx = f.shape[1]
        fny = f.shape[2]
        fnz = f.shape[3]
        self.nw = nw
        self.nx = nx
        self.ny = ny
        self.nz = nz
        cdef FourTensor ftens = FourTensor(fnw, fnx, fny, fnz, &f[0,0,0,0])
        self.scpp = new QuadcubicSpline(w0, dw, nw, x0, dx, nx, y0, dy, ny, z0, dz, nz, ftens, method)

    def coefficient(self, int h, int i, int j, int k, int nw, int nx, int ny, int nz):
        return self.scpp.getSplineCoefficient(h, i, j, k, nw, nx, ny, nz)
    
    def coefficients(self):
        cdef np.ndarray[ndim=4, dtype=np.float64_t, mode='c'] chijk = np.asarray(self.scpp.getSplineCoefficients().data()).reshape(self.nw, self.nx, self.ny, 256*self.nz)
        return chijk

    def eval(self, double w, double x, double y, double z):
        return self.scpp.evaluate(w, x, y, z)

    def deriv_w(self, double w, double x, double y, double z):
        return self.scpp.derivative_w(w, x, y, z)

    def deriv_x(self, double w, double x, double y, double z):
        return self.scpp.derivative_x(w, x, y, z)

    def deriv_y(self, double w, double x, double y, double z):
        return self.scpp.derivative_y(w, x, y, z)

    def deriv_z(self, double w, double x, double y, double z):
        return self.scpp.derivative_z(w, x, y, z)

    def deriv_ww(self, double w, double x, double y, double z):
        return self.scpp.derivative_ww(w, x, y, z)

    def deriv_xx(self, double w, double x, double y, double z):
        return self.scpp.derivative_xx(w, x, y, z)

    def deriv_yy(self, double w, double x, double y, double z):
        return self.scpp.derivative_yy(w, x, y, z)

    def deriv_zz(self, double w, double x, double y, double z):
        return self.scpp.derivative_zz(w, x, y, z)

    def deriv_wx(self, double w, double x, double y, double z):
        return self.scpp.derivative_wx(w, x, y, z)

    def deriv_wy(self, double w, double x, double y, double z):
        return self.scpp.derivative_wy(w, x, y, z)

    def deriv_wz(self, double w, double x, double y, double z):
        return self.scpp.derivative_wz(w, x, y, z)

    def deriv_xy(self, double w, double x, double y, double z):
        return self.scpp.derivative_xy(w, x, y, z)

    def deriv_xz(self, double w, double x, double y, double z):
        return self.scpp.derivative_xz(w, x, y, z)

    def deriv_yz(self, double w, double x, double y, double z):
        return self.scpp.derivative_yz(w, x, y, z)

