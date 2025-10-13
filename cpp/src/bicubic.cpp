#include "bicubic.hpp"
#include <iostream>

#define ENDPOINT_TOL 1.e-10

//////////////////////////////////////////////////////////////////
//////////////          BicubicSpline       ////////////////
//////////////////////////////////////////////////////////////////

BicubicSpline::BicubicSpline(const Vector &x, const Vector &y, Matrix &z, int method): BicubicSpline(x[0], x[1] - x[0], x.size() - 1, y[0], y[1] - y[0], y.size() - 1, z) {}
BicubicSpline::BicubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, Matrix &z, int method): dx(dx), dy(dy), nx(nx), ny(ny), x0(x0), y0(y0), cij(nx, 16*ny) {
	if(nx + 1 != z.rows() && ny + 1 != z.cols()){
		if(nx + 1 == z.cols() && ny + 1 == z.rows()){
			// switch x and y
			cij.transposeInPlace();
			computeSplineCoefficients(z, method);
		}else if((nx + 1)*(ny + 1) == z.size()){
			Matrix m_z = z.reshaped(ny + 1, nx + 1).transpose();
			computeSplineCoefficients(m_z, method);
		}else{
			std::cout << "ERROR: Indices of vectors and matrices do not match \n";
		}
	}else{
		computeSplineCoefficients(z, method);
	}
}

BicubicSpline::BicubicSpline(const Vector &x, const Vector &y, const Vector &z, int method): BicubicSpline(x[0], x[1] - x[0], x.size() - 1, y[0], y[1] - y[0], y.size() - 1, z) {}
BicubicSpline::BicubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, const Vector &z_vec, int method): dx(dx), dy(dy), nx(nx), ny(ny), x0(x0), y0(y0), cij(nx, 16*ny) {
	Matrix z(nx+1, ny+1, z_vec);
	if(nx + 1 != z.rows() && ny + 1 != z.cols()){
		if(nx + 1 == z.cols() && ny + 1 == z.rows()){
			// switch x and y
			cij.transposeInPlace();
			computeSplineCoefficients(z, method);
		}else if((nx + 1)*(ny + 1) == z.size()){
			Matrix m_z = z.reshaped(ny + 1, nx + 1).transpose();
			computeSplineCoefficients(m_z, method);
		}else{
			std::cout << "ERROR: Indices of vectors and matrices do not match \n";
		}
	}else{
		computeSplineCoefficients(z, method);
	}
}

double BicubicSpline::evaluate(const double x, const double y){
	int i = findXInterval(x);
	int j = findYInterval(y);
	return evaluateInterval(i, j, x, y);
}

double BicubicSpline::derivative_x(const double x, const double y){
	int i = findXInterval(x);
	int j = findYInterval(y);
	return evaluateDerivativeXInterval(i, j, x, y)/dx;
}

double BicubicSpline::derivative_y(const double x, const double y){
	int i = findXInterval(x);
	int j = findYInterval(y);
	return evaluateDerivativeYInterval(i, j, x, y)/dy;
}

double BicubicSpline::derivative_xy(const double x, const double y){
	int i = findXInterval(x);
	int j = findYInterval(y);
	return evaluateDerivativeXYInterval(i, j, x, y)/dx/dy;
}

double BicubicSpline::derivative_xx(const double x, const double y){
	int i = findXInterval(x);
	int j = findYInterval(y);
	return evaluateDerivativeXXInterval(i, j, x, y)/dx/dx;
}

double BicubicSpline::derivative_yy(const double x, const double y){
	int i = findXInterval(x);
	int j = findYInterval(y);
	return evaluateDerivativeYYInterval(i, j, x, y)/dy/dy;
}

Matrix BicubicSpline::computeSplineCoefficientsDY(Matrix &m_z, int method){
	int Nx = m_z.rows();
	int Ny = m_z.cols();
	Matrix m_zdy(Nx, Ny);
	for(int i = 0; i < Nx; i++){
		Vector z_xi = m_z.row(i);
		CubicSpline f_xi = CubicSpline(y0, dy, z_xi, method);
		for(int j = 0; j < Ny; j++){
			m_zdy(i, j) = dy*f_xi.derivative(y0 + j*dy);
		}
	}
	return m_zdy; 
}

Matrix BicubicSpline::computeSplineCoefficientsDX(Matrix &m_z, int method){
	int Nx = m_z.rows();
	int Ny = m_z.cols();
	Matrix m_zdx(Nx, Ny);
	for(int j = 0; j < Ny; j++){
		Vector z_yj = m_z.col(j);
		CubicSpline f_yj = CubicSpline(x0, dx, z_yj, method);
		for(int i = 0; i < Nx; i++){
			m_zdx(i, j) = dx*f_yj.derivative(x0 + i*dx);
		}
	}
	return m_zdx; 
}

void BicubicSpline::computeSplineCoefficients(Matrix &m_z, int method){
	// StopWatch watch;

	Matrix lmat(4, 4, 0.);
	lmat(0, 0) = 1.;
	lmat(1, 2) = 1.;
	lmat(2, 0) = -3.;
	lmat(2, 1) = 3.;
	lmat(2, 2) = -2.;
	lmat(2, 3) = -1.;
	lmat(3, 0) = 2.;
	lmat(3, 1) = -2.;
	lmat(3, 2) = 1.;
	lmat(3, 3) = 1.;
	
	Matrix m_zdx = computeSplineCoefficientsDX(m_z, method);
	Matrix m_zdy = computeSplineCoefficientsDY(m_z, method);
	Matrix m_zdxdy = computeSplineCoefficientsDY(m_zdx, method);
	// Matrix m_zdxdy2 = computeSplineCoefficientsDX(m_zdy);

	// int Nx = m_z.rows();
	// int Ny = m_z.cols();
	// for(int i = 0; i < Nx; i++){
	// 	for(int j = 0; j < Ny; j++){
	// 		// if(j == 0){
	// 		// 	std::cout << "dx = " << m_zdx(i, j) << "\n";
	// 		// }
	// 		// if(i == 0){
	// 		// 	std::cout << "dy = " << m_zdy(i, j) << "\n";
	// 		// }
	// 		std::cout << m_zdxdy2(i, j) << "\n";
	// 		std::cout << m_zdxdy(i, j) << "\n";
	// 	}
	// }

	// now this part we just have to accept as being inefficient because we
	// are mixing rows and columns no matter what. The important thing is that
	// we will store the relevant cofficients close to one another in memory

	// watch.start();
	{
			for(int i = 0; i < nx; i++){
				for(int j = 0; j < ny; j++){
					Matrix dmat(4, 4);
					dmat(0, 0) = m_z(i, j); // f(0,0)
					dmat(0, 1) = m_z(i, j + 1); // f(0,1)
					dmat(0, 2) = m_zdy(i, j); // fy(0,0)
					dmat(0, 3) = m_zdy(i, j + 1); // fy(0,1)
					dmat(1, 0) = m_z(i + 1, j); // f(1,0)
					dmat(1, 1) = m_z(i + 1, j + 1); // f(1,1)
					dmat(1, 2) = m_zdy(i + 1, j); // fy(1,0)
					dmat(1, 3) = m_zdy(i + 1, j + 1); // fy(1,1)
					dmat(2, 0) = m_zdx(i, j); // fx(0,0)
					dmat(2, 1) = m_zdx(i, j + 1); // fx(0,1)
					dmat(2, 2) = m_zdxdy(i, j); // fxy(0,0)
					dmat(2, 3) = m_zdxdy(i, j + 1); // fxy(0,1)
					dmat(3, 0) = m_zdx(i + 1, j); // fx(1,0)
					dmat(3, 1) = m_zdx(i + 1, j + 1); // fx(1,1)
					dmat(3, 2) = m_zdxdy(i + 1, j); // fxy(1,0)
					dmat(3, 3) = m_zdxdy(i + 1, j + 1); // fxy(1,1)

					// this part is slow. Just lots of matrix multiplication
					Matrix Dmat(4, 4);
					for(int k = 0; k < 4; k++){
						for(int l = 0; l < 4; l++){
							for(int m = 0; m < 4; m++){
								Dmat(k, l) += dmat(k, m)*lmat(l, m); // need transpose of lmat
							}
						}
					}
					for(int k = 0; k < 4; k++){
						for(int l = 0; l < 4; l++){
							for(int m = 0; m < 4; m++){
								cij(i, 4*(4*j + k) + l) += lmat(k, m)*Dmat(m, l);
							}
						}
					}
				}
			}
	}
	// watch.stop();
	// watch.print();
	// watch.reset();
}

double BicubicSpline::getSplineCoefficient(int i, int j, int nx, int ny){
	return cij(i, 4*(4*j + nx) + ny);
}

double BicubicSpline::evaluateInterval(int i, int j, const double x, const double y){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zvec[4];
	double result;

	for(int k = 0; k < 4; k++){
		zvec[k] = cij(i, 4*(4*j + k) + 0) + ybar*(cij(i, 4*(4*j + k) + 1) + ybar*(cij(i, 4*(4*j + k) + 2) + cij(i, 4*(4*j + k) + 3)*ybar));
	}

	result = zvec[0] + xbar*(zvec[1] + xbar*(zvec[2] + zvec[3]*xbar));

	return result;
}

double BicubicSpline::evaluateDerivativeXInterval(int i, int j, const double x, const double y){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zvec[4];
	double result;

	for(int k = 0; k < 4; k++){
		zvec[k] = cij(i, 4*(4*j + k) + 0) + ybar*(cij(i, 4*(4*j + k) + 1) + ybar*(cij(i, 4*(4*j + k) + 2) + cij(i, 4*(4*j + k) + 3)*ybar));
	}

	result = (zvec[1] + xbar*(2.*zvec[2] + 3.*zvec[3]*xbar));

	return result;
}

double BicubicSpline::evaluateDerivativeYInterval(int i, int j, const double x, const double y){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zvec[4];
	double result;


	for(int k = 0; k < 4; k++){
		zvec[k] = cij(i, 4*(4*j + k) + 1) + ybar*(2.*cij(i, 4*(4*j + k) + 2) + 3.*cij(i, 4*(4*j + k) + 3)*ybar);
	}

	result = zvec[0] + xbar*(zvec[1] + xbar*(zvec[2] + zvec[3]*xbar));
	
	return result;
}

double BicubicSpline::evaluateDerivativeXYInterval(int i, int j, const double x, const double y){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zvec[4];
	double result;

	for(int k = 0; k < 4; k++){
		zvec[k] = (cij(i, 4*(4*j + k) + 1) + ybar*(2.*cij(i, 4*(4*j + k) + 2) + 3.*cij(i, 4*(4*j + k) + 3)*ybar));
	}

	result = (zvec[1] + xbar*(2.*zvec[2] + 3.*zvec[3]*xbar));
	
	return result;
}

double BicubicSpline::evaluateDerivativeXXInterval(int i, int j, const double x, const double y){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zvec[4];
	double result;

	for(int k = 0; k < 4; k++){
		zvec[k] = cij(i, 4*(4*j + k) + 0) + ybar*(cij(i, 4*(4*j + k) + 1) + ybar*(cij(i, 4*(4*j + k) + 2) + cij(i, 4*(4*j + k) + 3)*ybar));
	}

	result = 2.*(zvec[2] + 3.*zvec[3]*xbar);

	return result;
}

double BicubicSpline::evaluateDerivativeYYInterval(int i, int j, const double x, const double y){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zvec[4];
	double result;

	for(int k = 0; k < 4; k++){
		zvec[k] = 2.*(cij(i, 4*(4*j + k) + 2) + 3.*cij(i, 4*(4*j + k) + 3)*ybar);
	}

	result = zvec[0] + xbar*(zvec[1] + xbar*(zvec[2] + zvec[3]*xbar));
	
	return result;
}

int BicubicSpline::findXInterval(const double x){
	int i = static_cast<int>((x-x0)/dx);
    if(i >= nx){
        return nx - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

int BicubicSpline::findYInterval(const double y){
	int i = static_cast<int>((y-y0)/dy);
    if(i >= ny){
        return ny - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

CubicSpline BicubicSpline::reduce_x(const double x){
    int i = findXInterval(x);
    double xbar = (x - x0 - i*dx)/dx;

	Matrix cubicCij(ny, 4);
	double xvec[4] = {1., xbar, xbar*xbar, xbar*xbar*xbar};

	// zj = xi*cij
	for(int j = 0; j < ny; j++){
		for(int k = 0; k < 4; k++){
			for(int l = 0; l < 4; l++){
				cubicCij(j, k) += cij(i, 16*j + 4*l + k)*xvec[l];
			}
		}
	}

    return CubicSpline(y0, dy, ny, cubicCij);
}

CubicSpline BicubicSpline::reduce_y(const double y){
    int j = findYInterval(y);
    double ybar = (y - y0 - j*dy)/dy;
	Matrix cubicCij(nx, 4);
	double yvec[4] = {1., ybar, ybar*ybar, ybar*ybar*ybar};

	// zj = xi*cij
	for(int i = 0; i < nx; i++){
		for(int k = 0; k < 4; k++){
			for(int l = 0; l < 4; l++){
				cubicCij(i, k) += cij(i, 4*(4*j + k) + l)*yvec[l];
			}
		}
	}

    return CubicSpline(x0, dx, nx, cubicCij);
}
