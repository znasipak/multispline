#include "quadcubic.hpp"
#include <iostream>

#define ENDPOINT_TOL 1.e-10

//////////////////////////////////////////////////////////////////
//////////////          QuadcubicSpline       ////////////////
//////////////////////////////////////////////////////////////////

QuadcubicSpline::QuadcubicSpline(const Vector &w, const Vector &x, const Vector &y, const Vector &z, FourTensor &f, int method): QuadcubicSpline(w[0], w[1] - w[0], w.size() - 1, x[0], x[1] - x[0], x.size() - 1, y[0], y[1] - y[0], y.size() - 1, z[0], z[1] - z[0], z.size() - 1, f, method) {}
QuadcubicSpline::QuadcubicSpline(double w0, double dw, int nw, double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, FourTensor &f, int method): dw(dw), dx(dx), dy(dy), dz(dz), nw(nw), nx(nx), ny(ny), nz(nz), w0(w0), x0(x0), y0(y0), z0(z0), cijk(nw, nx, ny, 256*nz) {
	if(nw + 1 == f.dim(0) && nx + 1 == f.dim(1) && ny + 1 == f.dim(2) && nz + 1 == f.dim(3)){
		computeSplineCoefficients(f, method);
	}else if(nw == f.dim(0) && nx == f.dim(1) && ny == f.dim(2) && 256*nz == f.dim(3)){
		cijk = f;
	}else{
		std::cout << "ERROR: Indices of vectors and matrices do not match \n";
	}
}

QuadcubicSpline::QuadcubicSpline(const Vector &w, const Vector &x, const Vector &y, const Vector &z, const Vector &f, int method): QuadcubicSpline(w[0], w[1] - w[0], w.size() - 1, x[0], x[1] - x[0], x.size() - 1, y[0], y[1] - y[0], y.size() - 1, z[0], z[1] - z[0], z.size() - 1, f, method) {}
QuadcubicSpline::QuadcubicSpline(double w0, double dw, int nw, double x0, double dx, int nx, double y0, double dy, int ny, double z0, double dz, int nz, const Vector &f_vec, int method): dw(dw), dx(dx), dy(dy), dz(dz), nw(nw), nx(nx), ny(ny), nz(nz), w0(w0), x0(x0), y0(y0), z0(z0), cijk(nw, nx, ny, 256*nz) {
	// std::cout << "In QuadcubicSpline constructor 2 \n";
	FourTensor f(nw+1, nx+1, ny+1, nz+1, f_vec);
	if(nw + 1 == f.dim(0) && nx + 1 == f.dim(1) && ny + 1 == f.dim(2) && nz + 1 == f.dim(3)){
		computeSplineCoefficients(f, method);
	}else{
		std::cout << "ERROR: Indices of vectors and matrices do not match \n";
	}
}

double QuadcubicSpline::evaluate(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateInterval(h, i, j, k, w, x, y, z);
}

double QuadcubicSpline::derivative_w(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeWInterval(h, i, j, k, w, x, y, z)/dw;
}

double QuadcubicSpline::derivative_x(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeXInterval(h, i, j, k, w, x, y, z)/dx;
}

double QuadcubicSpline::derivative_y(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeYInterval(h, i, j, k, w, x, y, z)/dy;
}

double QuadcubicSpline::derivative_z(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeZInterval(h, i, j, k, w, x, y, z)/dz;
}

double QuadcubicSpline::derivative_ww(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeWWInterval(h, i, j, k, w, x, y, z)/dw/dw;
}

double QuadcubicSpline::derivative_wx(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeWXInterval(h, i, j, k, w, x, y, z)/dw/dx;
}

double QuadcubicSpline::derivative_wy(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeWYInterval(h, i, j, k, w, x, y, z)/dw/dy;
}

double QuadcubicSpline::derivative_wz(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeWZInterval(h, i, j, k, w, x, y, z)/dw/dz;
}

double QuadcubicSpline::derivative_xy(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeXYInterval(h, i, j, k, w, x, y, z)/dx/dy;
}

double QuadcubicSpline::derivative_xz(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeXZInterval(h, i, j, k, w, x, y, z)/dx/dz;
}

double QuadcubicSpline::derivative_yz(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeYZInterval(h, i, j, k, w, x, y, z)/dz/dy;
}

double QuadcubicSpline::derivative_xx(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeXXInterval(h, i, j, k, w, x, y, z)/dx/dx;
}

double QuadcubicSpline::derivative_yy(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeYYInterval(h, i, j, k, w, x, y, z)/dy/dy;
}

double QuadcubicSpline::derivative_zz(const double w, const double x, const double y, const double z){
	int h = findWInterval(w);
	int i = findXInterval(x);
	int j = findYInterval(y);
	int k = findZInterval(z);
	return evaluateDerivativeZZInterval(h, i, j, k, w, x, y, z)/dz/dz;
}

double QuadcubicSpline::getSplineCoefficient(int h, int i, int j, int k, int nw, int nx, int ny, int nz){
	return cijk(h, i, j, 4*(4*(4*(4*k + nw) + nx) + ny) + nz);
}

void QuadcubicSpline::setSplineCoefficient(int h, int i, int j, int k, int nw, int nx, int ny, int nz, double coeff){
	cijk(h, i, j, 4*(4*(4*(4*k + nw) + nx) + ny) + nz) = coeff;
}

void QuadcubicSpline::computeSplineCoefficients(FourTensor &f, int method){
	ThreeTensor z(nx+1, ny + 1, nz + 1);
	std::vector<FourTensor> cijs(64, FourTensor(nx, ny, nz, nw + 1));
	for(int h = 0; h < nw + 1; h++){
		z = f.slice_w(h);
		TricubicSpline tspl(x0, dx, nx, y0, dy, ny, z0, dz, nz, z, method);
		for(int i = 0; i < nx; i++){
			for(int j = 0; j < ny; j++){
				for(int k = 0; k < nz; k++){
					for(int hx = 0; hx < 4; hx++){
						for(int hy = 0; hy < 4; hy++){
							for(int hz = 0; hz < 4; hz++){
								cijs[4*(4*hx + hy) + hz](i, j, k, h) = tspl.getSplineCoefficient(i, j, k, hx, hy, hz);
							}
						}
					}
				}
			}
		}
	}

	Vector fw(nw + 1);
	for(int i = 0; i < nx; i++){
		for(int j = 0; j < ny; j++){
			for(int k = 0; k < nz; k++){
				for(int hx = 0; hx < 4; hx++){
					for(int hy = 0; hy < 4; hy++){
						for(int hz = 0; hz < 4; hz++){
							fw = cijs[4*(4*hx + hy) + hz].slice_wxy(i, j, k);
							CubicSpline spl(w0, dw, nw, fw, method);
							for(int h = 0; h < nw; h++){
								for(int hw = 0; hw < 4; hw++){
									setSplineCoefficient(h, i, j, k, hw, hx, hy, hz, spl.getSplineCoefficient(h, hw));
								}
							}
						}
					}
				}
			}
		}
	}
}

double QuadcubicSpline::evaluateInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeWInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 3; nn++){
		result = result*wbar + (3. - nn)*xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeXInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 3; nn++){
			xvec[l] = xvec[l]*xbar + (3. - nn)*yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeYInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 3; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + (3. - nn)*zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 3; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + (3. - nn)*getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeWWInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 2; nn++){
		result = result*wbar + (3. - nn)*(2. - nn)*xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeWXInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 3; nn++){
			xvec[l] = xvec[l]*xbar + (3. - nn)*yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 3; nn++){
		result = result*wbar + (3. - nn)*xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeWYInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 3; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + (3. - nn)*zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 3; nn++){
		result = result*wbar +(3. - nn)*xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeWZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 3; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + (3. - nn)*getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 3; nn++){
		result = result*wbar + (3. - nn)*xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeXYInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 3; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + (3. - nn)*zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 3; nn++){
			xvec[l] = xvec[l]*xbar + (3. - nn)*yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeXZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 3; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + (3. - nn)*getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 3; nn++){
			xvec[l] = xvec[l]*xbar + (3. - nn)*yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeYZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 3; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + (3. - nn)*getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 3; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + (3. - nn)*zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeXXInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 2; nn++){
			xvec[l] = xvec[l]*xbar + (3. - nn)*(2. - nn)*yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeYYInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 4; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 2; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + (3. - nn)*(2. - nn)*zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

double QuadcubicSpline::evaluateDerivativeZZInterval(int h, int i, int j, int k, const double w, const double x, const double y, const double z){
	double xbar = (x - x0 - i*dx)/dx;
	double ybar = (y - y0 - j*dy)/dy;
	double zbar = (z - z0 - k*dz)/dz;
	double wbar = (w - w0 - h*dw)/dw;
	double zvec[64];
	double yvec[16];
	double xvec[4];
	double result = 0.;

	for(int l = 0; l < 4; l++){
		for(int m = 0; m < 4; m++){
			for(int n = 0; n < 4; n++){
				zvec[4*(4*l + m) + n] = 0.;
				for(int nn = 0; nn < 2; nn++){
					zvec[4*(4*l + m) + n] = zvec[4*(4*l + m) + n]*zbar + (3. - nn)*(2. - nn)*getSplineCoefficient(h, i, j, k, l, m, n, 3 - nn);
				}
			}
			yvec[4*l + m] = 0.;
			for(int nn = 0; nn < 4; nn++){
				yvec[4*l + m] = yvec[4*l + m]*ybar + zvec[4*(4*l + m) + (3 - nn)];
			}
		}
		xvec[l] = 0.;
		for(int nn = 0; nn < 4; nn++){
			xvec[l] = xvec[l]*xbar + yvec[4*l + (3 - nn)];
		}
	}
	for(int nn = 0; nn < 4; nn++){
		result = result*wbar + xvec[3 - nn];
	}

	return result;
}

int QuadcubicSpline::findWInterval(const double w){
	int i = static_cast<int>((w-w0)/dw);
    if(i >= nw){
        return nw - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

int QuadcubicSpline::findXInterval(const double x){
	int i = static_cast<int>((x-x0)/dx);
    if(i >= nx){
        return nx - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

int QuadcubicSpline::findYInterval(const double y){
	int i = static_cast<int>((y-y0)/dy);
    if(i >= ny){
        return ny - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

int QuadcubicSpline::findZInterval(const double z){
	int i = static_cast<int>((z-z0)/dz);
    if(i >= nz){
        return nz - 1;
    }
	if( i < 0){
		return 0;
	}
	return i;
}

FourTensor QuadcubicSpline::getSplineCoefficients(){
	return cijk;
}