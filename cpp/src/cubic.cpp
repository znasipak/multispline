#include "cubic.hpp"
#include <iostream>

#define ENDPOINT_TOL 1.e-10


//////////////////////////////////////////////////////////////////
//////////////           CubicSpline        ////////////////
//////////////////////////////////////////////////////////////////

CubicSpline::CubicSpline(double x0, double dx, const Vector &y, int method): CubicSpline(x0, dx, y.size()-1, y, method) {}
CubicSpline::CubicSpline(double x0, double dx, int nx, const Vector &y, int method): dx(dx), nintervals(nx), x0(x0), cij(nx, 4) {
	if(method == 0){
		computeSplineCoefficients(dx, y);
	}else if(method == 1){
		computeSplineCoefficientsNotAKnot(dx, y);
	}else if(method == 2){
		computeSplineCoefficientsZeroClamped(dx, y);
	}else if(method == 3){
		computeSplineCoefficientsE3(dx, y);
	}else if(method == 4){
		computeSplineCoefficientsNaturalFirst(dx, y);
	}else{
		computeSplineCoefficientsNotAKnot(dx, y);
	}
}

CubicSpline::CubicSpline(const Vector &x, const Vector &y, int method): dx(x[1] - x[0]), nintervals(x.size() - 1), x0(x[0]), cij(x.size() - 1, 4) {
	if(x.size() != y.size()){
		std::cout << "ERROR: Size of x and y vectors do not match \n";
	}
	if(method == 0){
		computeSplineCoefficients(dx, y);
	}else if(method == 1){
		computeSplineCoefficientsNotAKnot(dx, y);
	}else if(method == 2){
		computeSplineCoefficientsZeroClamped(dx, y);
	}else if(method == 3){
		computeSplineCoefficientsE3(dx, y);
	}else if(method == 4){
		computeSplineCoefficientsNaturalFirst(dx, y);
	}else{
		computeSplineCoefficientsNotAKnot(dx, y);
	}
}

CubicSpline::CubicSpline(double x0, double dx, int nx, Matrix cij): dx(dx), nintervals(nx), x0(x0), cij(cij) {}

double CubicSpline::getSplineCoefficient(int i, int j){
	return cij(i, j)*std::pow(dx, j);
}

double CubicSpline::evaluate(const double x){
	int i = findInterval(x);
	return evaluateInterval(i, x);
}

double CubicSpline::derivative(const double x){
	int i = findInterval(x);
	return evaluateDerivativeInterval(i, x);
}

double CubicSpline::derivative2(const double x){
	int i = findInterval(x);
	return evaluateSecondDerivativeInterval(i, x);
}

void CubicSpline::computeSplineCoefficients(double dx, const Vector &y){
	// Calculation with natural boundary conditions that follows the GSL algorithm
	// Essentially we first calculate the second-derivatives assuming y''(x0) = y''(xn) = 0
	// Then from the values of y and y'', we construct the spline coefficients

	int nsize = y.size() - 2;
	// Forward sweep method for tridiagonal solve on Wikipedia
	double a = 1.;
	Vector b(nsize, 4.);
	double c = 1.;
	Vector d(nsize);
	Vector ydx2Over2(nsize);

	for(int i = 1; i <= nsize; i++){
		d[i - 1] = 3.*(y[i + 1] - 2.0*y[i] + y[i - 1])/(dx*dx);
	}

	//forward sweep
	double w = 0.;
	for(int i = 2; i <= nsize; i++){
		w = a/b[i - 2];
		b[i - 1] = b[i - 1] - w*c;
		d[i - 1] = d[i - 1] - w*d[i - 2];
	}
	// back substitution
	ydx2Over2[nsize - 1] = d[nsize - 1]/b[nsize - 1];
	for(int i = 1; i < nsize; i++){
		ydx2Over2[nsize - i - 1] = (d[nsize - i - 1] - c*ydx2Over2[nsize - i])/b[nsize - i - 1];
	}

	int i = 0;
	cij(i, 0) = y[i];
	cij(i, 2) = 0;
	cij(i, 1) = (y[i + 1] - y[i])/dx - dx*(ydx2Over2[i])/3.0;
	cij(i, 3) = (ydx2Over2[i])/(3.0*dx);
	
	for(i = 1; i < nintervals - 1; i++){
		cij(i, 0) = y[i];
		cij(i, 2) = ydx2Over2[i - 1];
		cij(i, 1) = (y[i + 1] - y[i])/dx - dx*(ydx2Over2[i] + 2.*ydx2Over2[i - 1])/3.0;
		cij(i, 3) = (ydx2Over2[i] - ydx2Over2[i - 1])/(3.0*dx);
	}

	i = nintervals - 1;
	cij(i, 0) = y[i];
	cij(i, 2) = ydx2Over2[i - 1];
	cij(i, 1) = (y[i + 1] - y[i])/dx - dx*(2.*ydx2Over2[i - 1])/3.0;
	cij(i, 3) = -(ydx2Over2[i - 1])/(3.0*dx);
}

void CubicSpline::computeSplineCoefficientsNaturalFirst(double dx, const Vector &y){
	int nsize = y.size();
	// Forward sweep method for tridiagonal solve on Wikipedia
	Vector a(nsize, 1.);
	Vector b(nsize, 4.);
	Vector c(nsize, 1.);
	Vector d(nsize);
	Vector ydx(nsize);

	b[0] = 2.;
	c[0] = 1.;  
	b[nsize - 1] = 2.;
	a[nsize - 1] = 1.;

	for(int i = 1; i < nsize - 1; i++){
		d[i] = 3.*(y[i + 1] - y[i - 1])/(dx);
	}
	d[0] = 3.*(y[1] - y[0])/dx;
	d[nsize - 1] = 3.*(y[nsize - 1] - y[nsize - 2])/dx;

	//forward sweep
	double w = 0.;
	double temp = 0.;
	for(int i = 1; i < nsize; i++){
		w = a[i]/b[i - 1];
		temp = b[i] - w*c[i - 1];
		b[i] = temp;
		temp = d[i] - w*d[i - 1];
		d[i] = temp;
	}
	// back substitution
	ydx[nsize - 1] = d[nsize - 1]/b[nsize - 1];
	for(int i = 1; i < nsize; i++){
		ydx[nsize - i - 1] = (d[nsize - i - 1] - c[nsize - i - 1]*ydx[nsize - i])/b[nsize - i - 1];
	}
	
	for(int i = 0; i < nintervals; i++){
		cij(i, 0) = y[i];
		cij(i, 1) = ydx[i];
		cij(i, 2) = 3.*(y[i + 1] - y[i])/(dx*dx) - (ydx[i + 1] + 2.*ydx[i])/dx;
		cij(i, 3) = (ydx[i + 1] - ydx[i])/(3.*dx*dx) - 2.*cij(i, 2)/(3.*dx);
	}
}

void CubicSpline::computeSplineCoefficientsZeroClamped(double dx, const Vector &y){

	int nsize = y.size();
	// Forward sweep method for tridiagonal solve on Wikipedia
	Vector a(nsize, 1.);
	Vector b(nsize, 4.);
	Vector c(nsize, 1.);
	Vector d(nsize);
	Vector ydx(nsize);

	b[0] = 1.;
	c[0] = 0.;  
	b[nsize - 1] = 1.;
	a[nsize - 1] = 0.;

	for(int i = 1; i < nsize - 1; i++){
		d[i] = 3.*(y[i + 1] - y[i - 1])/(dx);
	}
	d[0] = 0.;
	d[nsize - 1] = 0.;

	//forward sweep
	double w = 0.;
	for(int i = 1; i < nsize; i++){
		w = a[i]/b[i - 1];
		b[i] = b[i] - w*c[i - 1];
		d[i] = d[i] - w*d[i - 1];
	}
	// back substitution
	ydx[nsize - 1] = d[nsize - 1]/b[nsize - 1];
	for(int i = 1; i < nsize; i++){
		ydx[nsize - i - 1] = (d[nsize - i - 1] - c[nsize - i - 1]*ydx[nsize - i])/b[nsize - i - 1];
	}
	
	for(int i = 0; i < nintervals; i++){
		cij(i, 0) = y[i];
		cij(i, 1) = ydx[i];
		cij(i, 2) = 3.*(y[i + 1] - y[i])/(dx*dx) - (ydx[i + 1] + 2.*ydx[i])/dx;
		cij(i, 3) = (ydx[i + 1] - ydx[i])/(3.*dx*dx) - 2.*cij(i, 2)/(3.*dx);
	}
}

void CubicSpline::computeSplineCoefficientsE3(double dx, const Vector &y){

	int nsize = y.size();
	// Forward sweep method for tridiagonal solve on Wikipedia
	Vector a(nsize, 1.);
	Vector b(nsize, 4.);
	Vector c(nsize, 1.);
	Vector d(nsize);
	Vector ydx(nsize);

	b[0] = 6.;
	c[0] = 18.;  
	b[nsize - 1] = 6.;
	a[nsize - 1] = 18.;

	for(int i = 1; i < nsize - 1; i++){
		d[i] = 3.*(y[i + 1] - y[i - 1])/(dx);
	}
	d[0] = (-y[3] + 9.*y[2] + 9.*y[1] - 17.*y[0])/dx;
	d[nsize - 1] = (y[nsize - 1 - 3] - 9.*y[nsize - 1 - 2] - 9.*y[nsize - 1 - 1] + 17.*y[nsize - 1])/dx;

	//forward sweep
	double w = 0.;
	for(int i = 1; i < nsize; i++){
		w = a[i]/b[i - 1];
		b[i] = b[i] - w*c[i - 1];
		d[i] = d[i] - w*d[i - 1];
	}
	// back substitution
	ydx[nsize - 1] = d[nsize - 1]/b[nsize - 1];
	for(int i = 1; i < nsize; i++){
		ydx[nsize - i - 1] = (d[nsize - i - 1] - c[nsize - i - 1]*ydx[nsize - i])/b[nsize - i - 1];
	}
	
	for(int i = 0; i < nintervals; i++){
		cij(i, 0) = y[i];
		cij(i, 1) = ydx[i];
		cij(i, 2) = 3.*(y[i + 1] - y[i])/(dx*dx) - (ydx[i + 1] + 2.*ydx[i])/dx;
		cij(i, 3) = (ydx[i + 1] - ydx[i])/(3.*dx*dx) - 2.*cij(i, 2)/(3.*dx);
	}
}

void CubicSpline::computeSplineCoefficientsNotAKnot(double dx, const Vector &y){
	// Calculation with not-a-knot boundary conditions that follows the GSL algorithm
	// Essentially we calculate the first-derivatives assuming y'''(x0) = y'''(x1) and y'''(xn) = y'''(xn-1)
	// Then from the values of y and y', we construct the spline coefficients

	int nsize = y.size();
	// Forward sweep method for tridiagonal solve on Wikipedia
	Vector a(nsize, 1.);
	Vector b(nsize, 4.);
	Vector c(nsize, 1.);
	Vector d(nsize);
	Vector ydx(nsize);

	b[0] = 2.;
	c[0] = 4.;  
	b[nsize - 1] = 2.;
	a[nsize - 1] = 4.;

	for(int i = 1; i < nsize - 1; i++){
		d[i] = 3.*(y[i + 1] - y[i - 1])/(dx);
	}
	// d[0] = 3.*(y[1] - y[0])/dx;
	// d[nsize - 1] = 3.*(y[nsize - 1] - y[nsize - 2])/dx;
	d[0] = (y[2] + 4.*y[1] - 5.*y[0])/dx;
	d[nsize - 1] = (5.*y[nsize - 1] - 4.*y[nsize - 2] - y[nsize - 3])/dx;

	//forward sweep
	double w = 0.;
	for(int i = 1; i < nsize; i++){
		w = a[i]/b[i - 1];
		b[i] = b[i] - w*c[i - 1];
		d[i] = d[i] - w*d[i - 1];
	}
	// back substitution
	ydx[nsize - 1] = d[nsize - 1]/b[nsize - 1];
	for(int i = 1; i < nsize; i++){
		ydx[nsize - i - 1] = (d[nsize - i - 1] - c[nsize - i - 1]*ydx[nsize - i])/b[nsize - i - 1];
	}
	
	for(int i = 0; i < nintervals; i++){
		cij(i, 0) = y[i];
		cij(i, 1) = ydx[i];
		cij(i, 2) = 3.*(y[i + 1] - y[i])/(dx*dx) - (ydx[i + 1] + 2.*ydx[i])/dx;
		cij(i, 3) = (ydx[i + 1] - ydx[i])/(3.*dx*dx) - 2.*cij(i, 2)/(3.*dx);
	}
}

double CubicSpline::evaluateInterval(int i, const double x){
	double xbar = (x - x0 - i*dx);
	return cij(i, 0) + xbar*(cij(i, 1) + xbar*(cij(i, 2) + cij(i, 3)*xbar));
}

double CubicSpline::evaluateDerivativeInterval(int i, const double x){
	double xbar = (x - x0 - i*dx);
	return cij(i, 1) + xbar*(2.0*cij(i, 2) + 3.0*cij(i, 3)*xbar);
}

double CubicSpline::evaluateSecondDerivativeInterval(int i, const double x){
	double xbar = (x - x0 - i*dx);
	return 2.0*(cij(i, 2) + 3.0*cij(i, 3)*xbar);
}

int CubicSpline::findInterval(const double x){
	int i = static_cast<int>((x-x0)/dx);
    if(i >= nintervals){
        return nintervals - 1;
    }
	if(i < 0){
		return 0;
	}
	return i;
}
