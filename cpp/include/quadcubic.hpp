#ifndef QUADCUBIC_HPP
#define QUADCUBIC_HPP

#include "tricubic.hpp"

/////////////////////////////////////////////////////////
////               Basic Interpolators               ////
/////////////////////////////////////////////////////////

class QuadcubicSpline{
public:
	QuadcubicSpline(const Vector &w, const Vector &x, const Vector &y, const Vector &z, FourTensor &f, int method = 3);
	QuadcubicSpline(double w0, double dw, int nw, double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, FourTensor &f, int method = 3);
	QuadcubicSpline(const Vector &w, const Vector &x, const Vector &y, const Vector &z, const Vector &f, int method = 3);
	QuadcubicSpline(double w0, double dw, int nw, double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, const Vector &f_vec, int method = 3);
	double evaluate(const double w, const double x, const double y, const double z);
    double derivative_w(const double w, const double x, const double y, const double z);
	double derivative_x(const double w, const double x, const double y, const double z);
    double derivative_y(const double w, const double x, const double y, const double z);
	double derivative_z(const double w, const double x, const double y, const double z);
	double derivative_wx(const double w, const double x, const double y, const double z);
	double derivative_wy(const double w, const double x, const double y, const double z);
	double derivative_wz(const double w, const double x, const double y, const double z);
    double derivative_xy(const double w, const double x, const double y, const double z);
	double derivative_xz(const double w, const double x, const double y, const double z);
	double derivative_yz(const double w, const double x, const double y, const double z);
	double derivative_ww(const double w, const double x, const double y, const double z);
    double derivative_xx(const double w, const double x, const double y, const double z);
    double derivative_yy(const double w, const double x, const double y, const double z);
	double derivative_zz(const double w, const double x, const double y, const double z);

	double getSplineCoefficient(int h, int i, int j, int k, int nw, int nx, int ny, int nz);
	FourTensor getSplineCoefficients();

private:
	double evaluateInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);

    double evaluateDerivativeWInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	double evaluateDerivativeXInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
    double evaluateDerivativeYInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	double evaluateDerivativeZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	
	double evaluateDerivativeWWInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	double evaluateDerivativeWXInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
    double evaluateDerivativeWYInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	double evaluateDerivativeWZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
    
	double evaluateDerivativeXXInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	double evaluateDerivativeXYInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	double evaluateDerivativeXZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	
	double evaluateDerivativeYYInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	double evaluateDerivativeYZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	
	double evaluateDerivativeZZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z);
	
	void computeSplineCoefficients(FourTensor &z, int method = 3);
	
	int findWInterval(const double w);
	int findXInterval(const double x);
	int findYInterval(const double y);
	int findZInterval(const double z);

	void setSplineCoefficient(int h, int i, int j, int k, int nw, int nx, int ny, int nz, double coeff);

	double dw;
	double dx;
	double dy;
	double dz;
	int nw;
	int nx;
	int ny;
	int nz;
	double w0;
	double x0;
	double y0;
	double z0;
	FourTensor cijk;
};

#endif