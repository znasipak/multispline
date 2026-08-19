#include "tricubic.hpp"
#include <iostream>

#define ENDPOINT_TOL 1.e-10

//////////////////////////////////////////////////////////////////
//////////////          TricubicSpline       ////////////////
//////////////////////////////////////////////////////////////////

TricubicSpline::TricubicSpline(const Vector &x, const Vector &y, const Vector &z, ThreeTensor &f, int method): TricubicSpline(x[0], x[1] - x[0], x.size() - 1, y[0], y[1] - y[0], y.size() - 1, z[0], z[1] - z[0], z.size() - 1, f, method) {}
TricubicSpline::TricubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, ThreeTensor &f, int method): dx(dx), dy(dy), dz(dz), nx(nx), ny(ny), nz(nz), x0(x0), y0(y0), z0(z0), cijk(nx, ny, 64*nz) {
	if(nx + 1 == f.rows() && ny + 1 == f.cols() && nz + 1 == f.slcs()){
		computeSplineCoefficients(f, method);
	}else if(nx == f.rows() && ny == f.cols() && 64*nz == f.slcs()){
		cijk = f;
	}else{
		std::cout << "ERROR: Indices of vectors and matrices do not match \n";
	}
}

TricubicSpline::TricubicSpline(const Vector &x, const Vector &y, const Vector &z, const Vector &f, int method): TricubicSpline(x[0], x[1] - x[0], x.size() - 1, y[0], y[1] - y[0], y.size() - 1, z[0], z[1] - z[0], z.size() - 1, f, method) {}
TricubicSpline::TricubicSpline(double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, const Vector &f_vec, int method): dx(dx), dy(dy), dz(dz), nx(nx), ny(ny), nz(nz), x0(x0), y0(y0), z0(z0), cijk(nx, ny, 64*nz) {
	ThreeTensor f(nx+1, ny+1, nz+1, f_vec);
	if(nx + 1 == f.rows() && ny + 1 == f.cols() && nz + 1 == f.slcs()){
		computeSplineCoefficients(f, method);
	}else{
		std::cout << "ERROR: Indices of vectors and matrices do not match \n";
	}
	// for(int i = 0; i < 4; i++){
	// 	std::cout << (getSplineCoefficient(0, 0, 0, 0, 0, i)) << "\n";
	// }
}

double TricubicSpline::evaluate(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateInterval(i, j, k, x, y, z);
}

double TricubicSpline::derivative_x(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeXInterval(i, j, k, x, y, z)/dx;
}

double TricubicSpline::derivative_y(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeYInterval(i, j, k, x, y, z)/dy;
}

double TricubicSpline::derivative_z(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeZInterval(i, j, k, x, y, z)/dz;
}

double TricubicSpline::derivative_xy(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeXYInterval(i, j, k, x, y, z)/dx/dy;
}

double TricubicSpline::derivative_xz(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeXZInterval(i, j, k, x, y, z)/dx/dz;
}

double TricubicSpline::derivative_yz(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeYZInterval(i, j, k, x, y, z)/dz/dy;
}

double TricubicSpline::derivative_xx(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeXXInterval(i, j, k, x, y, z)/dx/dx;
}

double TricubicSpline::derivative_yy(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeYYInterval(i, j, k, x, y, z)/dy/dy;
}

double TricubicSpline::derivative_zz(const double x, const double y, const double z){
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeZZInterval(i, j, k, x, y, z)/dz/dz;
}

double TricubicSpline::getSplineCoefficient(int i, int j, int k, int nx, int ny, int nz){
	return cijk(i, j, 4*(4*(4*k + nx) + ny) + nz);
}

void TricubicSpline::setSplineCoefficient(int i, int j, int k, int nx, int ny, int nz, double coeff){
	cijk(i, j, 4*(4*(4*k + nx) + ny) + nz) = coeff;
}

void TricubicSpline::computeSplineCoefficients(ThreeTensor &f, int method){
	Matrix z(ny + 1, nz + 1);
	std::vector<ThreeTensor> cijs(16, ThreeTensor(ny, nz, nx + 1));
	for(int i = 0; i < nx + 1; i++){
		z = f.row(i);
		BicubicSpline bspl(y0, dy, ny, z0, dz, nz, z, method);
		for(int j = 0; j < ny; j++){
			for(int k = 0; k < nz; k++){
				for(int iy = 0; iy < 4; iy++){
					for(int iz = 0; iz < 4; iz++){
						cijs[4*iy + iz](j, k, i) = bspl.getSplineCoefficient(j, k, iy, iz);
					}
				}
			}
		}
	}

	Vector fx(nx + 1);
	for(int j = 0; j < ny; j++){
		for(int k = 0; k < nz; k++){
			for(int iy = 0; iy < 4; iy++){
				for(int iz = 0; iz < 4; iz++){
					fx = cijs[4*iy + iz].rowcol(j, k);
					CubicSpline spl(x0, dx, nx, fx, method);
					for(int i = 0; i < nx; i++){
						for(int ix = 0; ix < 4; ix++){
							setSplineCoefficient(i, j, k, ix, iy, iz, spl.getSplineCoefficient(i, ix));
						}
					}
				}
			}
		}
	}
}

double TricubicSpline::evaluateInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			yvec[l] = yvec[l]*ybar + zvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*xbar + yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeXInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			yvec[l] = yvec[l]*ybar + zvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 3; nn++){
		result = result*xbar + (3. - nn)*yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeYInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 3; nn++){
			yvec[l] = yvec[l]*ybar + (3. - nn)*zvec[4*l + 3 - nn];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*xbar + yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeZInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 3; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + (3. - nn)*getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			yvec[l] = yvec[l]*ybar + zvec[4*l + 3 - nn];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*xbar + yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeXYInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 3; nn++){
			yvec[l] = yvec[l]*ybar + (3. - nn)*zvec[4*l + 3 - nn];
		}
	}
	for(int nn = 0; nn < 3; nn++){
		result = result*xbar + (3. - nn)*yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeXZInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 3; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + (3. - nn)*getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			yvec[l] = yvec[l]*ybar + zvec[4*l + 3 - nn];
		}
	}
	for(int nn = 0; nn < 3; nn++){
		result = result*xbar + (3. - nn)*yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeYZInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 3; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + (3. - nn)*getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 3; nn++){
			yvec[l] = yvec[l]*ybar + (3. - nn)*zvec[4*l + 3 - nn];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*xbar + yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeXXInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			yvec[l] = yvec[l]*ybar + zvec[4*l + 3 - nn];
		}
	}
	for(int nn = 0; nn < 2; nn++){
		result = result*xbar + (3. - nn)*(2. - nn)*yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeYYInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 2; nn++){
			yvec[l] = yvec[l]*ybar + (3. - nn)*(2. - nn)*zvec[4*l + 3 - nn];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*xbar + yvec[3 - nn];
	}

	return result;
}

double TricubicSpline::evaluateDerivativeZZInterval(int i, int j, int k, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double zvec[16];
	double yvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			// zvec[4*l + m] = getSplineCoefficient(i, j, k, l, m, 0) + zbar*(getSplineCoefficient(i, j, k, l, m, 1) + zbar*(getSplineCoefficient(i, j, k, l, m, 2) + getSplineCoefficient(i, j, k, l, m, 3)*zbar));
			zvec[4*l + m] = 0.;
			for(int nn = 0; nn < 2; nn++){
				zvec[4*l + m] = zvec[4*l + m]*zbar + (3. - nn)*(2. - nn)*getSplineCoefficient(i, j, k, l, m, 3 - nn);
			}
		}
		yvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			yvec[l] = yvec[l]*ybar + zvec[4*l + 3 - nn];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*xbar + yvec[3 - nn];
	}

	return result;
}

int TricubicSpline::findXInterval(const double x){
	int i = static_cast<int>((x-x0)/dx);
    if(i >= nx){
        return nx - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

int TricubicSpline::findYInterval(const double y){
	int i = static_cast<int>((y-y0)/dy);
    if(i >= ny){
        return ny - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

int TricubicSpline::findZInterval(const double z){
	int i = static_cast<int>((z-z0)/dz);
    if(i >= nz){
        return nz - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

ThreeTensor TricubicSpline::getSplineCoefficients(){
	return cijk;
}
