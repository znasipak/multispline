#ifndef TRICUBIC_HPP
#define TRICUBIC_HPP

#include "bicubic.hpp"

/////////////////////////////////////////////////////////
////               Basic Interpolators               ////
/////////////////////////////////////////////////////////

class TricubicSpline{
public:
	TricubicSpline(const Vector &x, const Vector &y, const Vector &z, ThreeTensor &f, int method = 3);
	TricubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, ThreeTensor &f, int method = 3);
	TricubicSpline(const Vector &x, const Vector &y, const Vector &z, const Vector &f, int method = 3);
	TricubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, const Vector &f_vec, int method = 3);
	double evaluate(const double x, const double y, const double z);
    double derivative_x(const double x, const double y, const double z);
    double derivative_y(const double x, const double y, const double z);
	double derivative_z(const double x, const double y, const double z);
    double derivative_xy(const double x, const double y, const double z);
	double derivative_xz(const double x, const double y, const double z);
	double derivative_yz(const double x, const double y, const double z);
    double derivative_xx(const double x, const double y, const double z);
    double derivative_yy(const double x, const double y, const double z);
	double derivative_zz(const double x, const double y, const double z);
    // BicubicSpline reduce_x(const double x);
    // BicubicSpline reduce_y(const double y);
	// BicubicSpline reduce_z(const double z);

	double getSplineCoefficient(int i, int j, int k, int nx, int ny, int nz);
	ThreeTensor getSplineCoefficients();

private:
	double evaluateInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeXInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeYInterval(int i, int j, int k, const double x, const double y, const double z);
	double evaluateDerivativeZInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeXYInterval(int i, int j, int k, const double x, const double y, const double z);
	double evaluateDerivativeXZInterval(int i, int j, int k, const double x, const double y, const double z);
	double evaluateDerivativeYZInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeXXInterval(int i, int j, int k, const double x, const double y, const double z);
    double evaluateDerivativeYYInterval(int i, int j, int k, const double x, const double y, const double z);
	double evaluateDerivativeZZInterval(int i, int j, int k, const double x, const double y, const double z);
	void computeSplineCoefficients(ThreeTensor &z, int method = 3);
	int findXInterval(const double x);
	int findYInterval(const double y);
	int findZInterval(const double z);

	void setSplineCoefficient(int i, int j, int k, int nx, int ny, int nz, double coeff);

	double dx;
	double dy;
	double dz;
	int nx;
	int ny;
	int nz;
	double x0;
	double y0;
	double z0;
	ThreeTensor cijk;
};

#endif