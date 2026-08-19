#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

class StopWatch{
public:
	StopWatch();

	void start();
	void stop();
	void reset();
	void print();
	void print(size_t cycles);
	double time();

private:
	double time_elapsed;
	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t2;
};

typedef std::vector<double> Vector;
class Matrix{
public:
	Matrix();
	Matrix(size_t n);
	Matrix(size_t n, size_t m);
	Matrix(size_t n, size_t m, Vector A);
	Matrix(size_t n, size_t m, double val);

	size_t rows() const;
	size_t cols() const;
	size_t size() const;

	void row_replace(size_t i, Vector row);
	void col_replace(size_t j, Vector col);

	Vector row(size_t i);
	Vector col(size_t j);

	void reshape(size_t n, size_t m);
	Matrix reshaped(size_t n, size_t m) const;
	Matrix transpose() const;
	void transposeInPlace();

	void set_value(size_t i, size_t j, double val);

	double& operator()(size_t i, size_t j);
	const double& operator()(size_t i, size_t j) const;

private:
	size_t _n;
	size_t _m;
	Vector _A;
};

class ThreeTensor{
public:
	ThreeTensor();
	ThreeTensor(size_t nx);
	ThreeTensor(size_t nx, size_t ny, size_t nz);
	ThreeTensor(size_t nx, size_t ny, size_t nz, Vector A);
	ThreeTensor(size_t nx, size_t ny, size_t nz, double *A);
	ThreeTensor(size_t nx, size_t ny, size_t nz, double val);

	size_t rows() const;
	size_t cols() const;
	size_t slcs() const; // slices
	size_t size() const;

	void row_replace(size_t i, Matrix row);
	void col_replace(size_t j, Matrix col);
	void slc_replace(size_t k, Matrix slc);

	Matrix row(size_t i);
	Vector rowcol(size_t i, size_t j);
	Vector rowslc(size_t i, size_t k);
	Matrix col(size_t j);
	Vector colslc(size_t j, size_t k);
	Matrix slc(size_t k);

	void reshape(size_t nx, size_t ny, size_t nz);
	ThreeTensor reshaped(size_t nx, size_t ny, size_t nz) const;

	void set_value(size_t i, size_t j, size_t k, double val);

	double& operator()(size_t i, size_t j, size_t k);
	const double& operator()(size_t i, size_t j, size_t k) const;

	Vector data();

private:
	size_t _nx;
	size_t _ny;
	size_t _nz;
	Vector _A;
};

class FourTensor{
	public:
		FourTensor();
		FourTensor(size_t nw);
		FourTensor(size_t nw, size_t nx, size_t ny, size_t nz);
		FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, Vector A);
		FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, double *A);
		FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, double val);
	
		size_t dim(size_t d) const;
		size_t size() const;
	
		// void slice_replace(size_t d, size_t i, ThreeTensor slice);
		// void slice_replace(size_t d1, size_t d2, size_t i, size_t j, Matrix slice);
		// void slice_replace(size_t d1, size_t d2, size_t d3, size_t i, size_t j, size_t k, Vector slice);
	
		ThreeTensor slice(size_t d, size_t i);
		ThreeTensor slice_w(size_t i);
		ThreeTensor slice_x(size_t i);
		ThreeTensor slice_y(size_t i);
		ThreeTensor slice_z(size_t i);
		
		Matrix slice(size_t d1, size_t d2, size_t i, size_t j);
		Matrix slice_wx(size_t i, size_t j);
		Matrix slice_wy(size_t i, size_t j);
		Matrix slice_wz(size_t i, size_t j);
		Matrix slice_xy(size_t i, size_t j);
		Matrix slice_xz(size_t i, size_t j);
		Matrix slice_yz(size_t i, size_t j);
		
		Vector slice(size_t d1, size_t d2, size_t d3, size_t i, size_t j, size_t k);
		Vector slice_wxy(size_t i, size_t j, size_t k);
		Vector slice_wxz(size_t i, size_t j, size_t k);
		Vector slice_wyz(size_t i, size_t j, size_t k);
		Vector slice_xyz(size_t i, size_t j, size_t k);
	
		void reshape(size_t nw, size_t nx, size_t ny, size_t nz);
		FourTensor reshaped(size_t nw, size_t nx, size_t ny, size_t nz) const;
	
		void set_value(size_t i, size_t j, size_t k, size_t l, double val);
	
		double& operator()(size_t i, size_t j, size_t k, size_t l);
		const double& operator()(size_t i, size_t j, size_t k, size_t l) const;
	
		Vector data();
	
	private:
		size_t _nw;
		size_t _nx;
		size_t _ny;
		size_t _nz;
		Vector _A;
	};

#endif