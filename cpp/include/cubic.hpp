#ifndef CUBIC_HPP
#define CUBIC_HPP

#include "utils.hpp"

/////////////////////////////////////////////////////////
////               Basic Interpolators               ////
/////////////////////////////////////////////////////////

class CubicSpline{
public:
	CubicSpline(double x0, double dx, int nx, const Vector &y, int method = 1);
	CubicSpline(double x0, double dx, const Vector &y, int method = 1);
	CubicSpline(const Vector &x, const Vector &y, int method = 1);

    CubicSpline(double x0, double dx, int nintervals, Matrix cij);
	
	double evaluate(const double x);
    double derivative(const double x);
    double derivative2(const double x);

	double getSplineCoefficient(int i, int j);

private:
	double evaluateInterval(int i, const double x);
    double evaluateDerivativeInterval(int i, const double x);
    double evaluateSecondDerivativeInterval(int i, const double x);
	void computeSplineCoefficients(double dx, const Vector &y);
	void computeSplineCoefficientsNaturalFirst(double dx, const Vector &y);
	void computeSplineCoefficientsNotAKnot(double dx, const Vector &y);
	void computeSplineCoefficientsZeroClamped(double dx, const Vector &y);
	void computeSplineCoefficientsE3(double dx, const Vector &y);
	int findInterval(const double x);

	double dx;
	int nintervals;
	double x0;
	Matrix cij;
};

#endif