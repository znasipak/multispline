#ifndef BICUBIC_HPP
#define BICUBIC_HPP

#include "utils.hpp"
#include "cubic.hpp"

/////////////////////////////////////////////////////////
////               Basic Interpolators               ////
/////////////////////////////////////////////////////////

class BicubicSpline{
public:
	BicubicSpline(const Vector &x, const Vector &y, Matrix &z, int method = 3);
	BicubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, Matrix &z, int method = 3);
	BicubicSpline(const Vector &x, const Vector &y, const Vector &z, int method = 3);
	BicubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, const Vector &z_vec, int method = 3);
	double evaluate(const double x, const double y);
    double derivative_x(const double x, const double y);
    double derivative_y(const double x, const double y);
    double derivative_xy(const double x, const double y);
    double derivative_xx(const double x, const double y);
    double derivative_yy(const double x, const double y);
    CubicSpline reduce_x(const double x);
    CubicSpline reduce_y(const double y);

	double getSplineCoefficient(int i, int j, int nx, int ny);

private:
	double evaluateInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeXInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeYInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeXYInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeXXInterval(int i, int j, const double x, const double y);
    double evaluateDerivativeYYInterval(int i, int j, const double x, const double y);
	Matrix computeSplineCoefficientsDX(Matrix &m_z, int method = 3);
	Matrix computeSplineCoefficientsDY(Matrix &m_z, int method = 3);
	void computeSplineCoefficients(Matrix &z, int method = 3);
	int findXInterval(const double x);
	int findYInterval(const double y);

	double dx;
	double dy;
	int nx;
	int ny;
	double x0;
	double y0;
	Matrix cij;
};

#endif