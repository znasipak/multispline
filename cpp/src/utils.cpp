#include "utils.hpp"
#include <iostream>

#define ENDPOINT_TOL 1.e-10

///////////////////////////////////////////////////////////////////
//////////////              Matrix Class           ////////////////
///////////////////////////////////////////////////////////////////

Matrix::Matrix() {}
Matrix::Matrix(size_t n): _n(n), _m(n), _A(n*n) {}
Matrix::Matrix(size_t n, size_t m): _n(n), _m(m), _A(n*m) {}
Matrix::Matrix(size_t n, size_t m, Vector A): _n(n), _m(m), _A(n*m) {
	if(A.size() == _A.size()){
		_A = A;
	}
}
Matrix::Matrix(size_t n, size_t m, double val): _n(n), _m(m), _A(n*m, val) {}

size_t Matrix::rows() const{
	return _n;
}
size_t Matrix::cols() const{
	return _m;
}
size_t Matrix::size() const{
	return _A.size();
}

void Matrix::row_replace(size_t i, Vector row){
	for(size_t j = 0; j < _m; j++){
		_A[i*_m + j] = row[j];
	}
}
void Matrix::col_replace(size_t i, Vector col){
	for(size_t j = 0; j < _n; j++){
		_A[j*_m + i] = col[j];
	}
}

Vector Matrix::row(size_t i){
	Vector row(_m);
	for(size_t j = 0; j < _m; j++){
		row[j] = _A[i*_m + j];
	}
	return row;
}
Vector Matrix::col(size_t i){
	Vector col(_n);
	for(size_t j = 0; j < _n; j++){
		col[j] = _A[j*_m + i];
	}
	return col;
}

void Matrix::reshape(size_t n, size_t m){
	_n = n;
	_m = m;
}

Matrix Matrix::reshaped(size_t n, size_t m) const{
	return Matrix(n, m , _A);
}

Matrix Matrix::transpose() const{
	Matrix AT(_m, _n);
	{
			for(size_t i = 0; i < _n; i++){
				for(size_t j = 0; j < _m; j++){
					AT(j, i) = _A[i*_m + j];
				}
			}
	}
	return AT;
}

void Matrix::transposeInPlace(){
	Vector AT(_A.size());
	{
		for(size_t i = 0; i < _n; i++){
			for(size_t j = 0; j < _m; j++){
				AT[j*_n + i] = _A[i*_m + j];
			}
		}

		for(size_t i = 0; i < _n*_m; i++){
			_A[i] = AT[i];
		}
	}
	size_t m = _n;
	_n = _m;
	_m = m;
}

void Matrix::set_value(size_t i, size_t j, double val){
	_A[i*_m + j] = val;
}

double& Matrix::operator()(size_t i, size_t j){
	return _A[i*_m + j];
}
const double& Matrix::operator()(size_t i, size_t j) const{
	return _A[i*_m + j];
}

//////////////////////////////////////////////////////////////////
//////////////           ThreeTensor Class        ////////////////
//////////////////////////////////////////////////////////////////

ThreeTensor::ThreeTensor() {}
ThreeTensor::ThreeTensor(size_t n): _nx(n), _ny(n), _nz(n), _A(n*n*n) {}
ThreeTensor::ThreeTensor(size_t nx, size_t ny, size_t nz): _nx(nx), _ny(ny), _nz(nz), _A(nx*ny*nz) {}
ThreeTensor::ThreeTensor(size_t nx, size_t ny, size_t nz, double *A): _nx(nx), _ny(ny), _nz(nz), _A(A, A + nx*ny*nz) {}
ThreeTensor::ThreeTensor(size_t nx, size_t ny, size_t nz, Vector A): _nx(nx), _ny(ny), _nz(nz), _A(nx*ny*nz) {
	if(A.size() == _A.size()){
		_A = A;
	}else{
		std::cout << "ERROR: Sizes do not match \n";
	}
}
ThreeTensor::ThreeTensor(size_t nx, size_t ny, size_t nz, double val): _nx(nx), _ny(ny), _nz(nz), _A(nx*ny*nz, val) {}

size_t ThreeTensor::rows() const{
	return _nx;
}
size_t ThreeTensor::cols() const{
	return _ny;
}
size_t ThreeTensor::slcs() const{
	return _nz;
}
size_t ThreeTensor::size() const{
	return _A.size();
}

void ThreeTensor::row_replace(size_t i, Matrix row){
	for(size_t j = 0; j < _ny; j++){
		for(size_t k = 0; k < _nz; k++){
			_A[(i*_ny + j)*_nz + k] = row(j, k);
		}
	}
}
void ThreeTensor::col_replace(size_t j, Matrix col){
	for(size_t i = 0; i < _nx; i++){
		for(size_t k = 0; k < _nz; k++){
			_A[(i*_ny + j)*_nz + k] = col(i, k);
		}
	}
}
void ThreeTensor::slc_replace(size_t k, Matrix slc){
	for(size_t i = 0; i < _nx; i++){
		for(size_t j = 0; j < _ny; j++){
			_A[(i*_ny + j)*_nz + k] = slc(i, j);
		}
	}
}

Matrix ThreeTensor::row(size_t i){
	Matrix row(_ny, _nz);
	for(size_t j = 0; j < _ny; j++){
		for(size_t k = 0; k < _nz; k++){
			row(j, k) = _A[(i*_ny + j)*_nz + k];
		}
	}
	return row;
}
Vector ThreeTensor::rowcol(size_t i, size_t j){
	Vector row(_nz);
	for(size_t k = 0; k < _nz; k++){
		row[k] = _A[(i*_ny + j)*_nz + k];
	}
	return row;
}
Vector ThreeTensor::rowslc(size_t i, size_t k){
	Vector row(_ny);
	for(size_t j = 0; j < _ny; j++){
		row[j] = _A[(i*_ny + j)*_nz + k];
	}
	return row;
}
Matrix ThreeTensor::col(size_t j){
	Matrix col(_nx, _nz);
	for(size_t i = 0; i < _nx; i++){
		for(size_t k = 0; k < _nz; k++){
			col(i, k) = _A[(i*_ny + j)*_nz + k];
		}
	}
	return col;
}
Vector ThreeTensor::colslc(size_t j, size_t k){
	Vector col(_nx);
	for(size_t i = 0; i < _nx; i++){
		col[i] = _A[(i*_ny + j)*_nz + k];
	}
	return col;
}
Matrix ThreeTensor::slc(size_t k){
	Matrix slc(_nx, _ny);
	for(size_t i = 0; i < _nx; i++){
		for(size_t j = 0; j < _ny; j++){
			slc(i, j) = _A[(i*_ny + j)*_nz + k];
		}
	}
	return slc;
}

void ThreeTensor::reshape(size_t nx, size_t ny, size_t nz){
	_nx = nx;
	_ny = ny;
	_nz = nz;
}

ThreeTensor ThreeTensor::reshaped(size_t nx, size_t ny, size_t nz) const{
	return ThreeTensor(nx, ny, nz, _A);
}

void ThreeTensor::set_value(size_t i, size_t j, size_t k, double val){
	_A[(i*_ny + j)*_nz + k] = val;
}

double& ThreeTensor::operator()(size_t i, size_t j, size_t k){
	return _A[(i*_ny + j)*_nz + k];
}
const double& ThreeTensor::operator()(size_t i, size_t j, size_t k) const{
	return _A[(i*_ny + j)*_nz + k];
}

Vector ThreeTensor::data(){
	return _A;
}

//////////////////////////////////////////////////////////////////
//////////////           FourTensor Class        ////////////////
//////////////////////////////////////////////////////////////////

FourTensor::FourTensor() {}
FourTensor::FourTensor(size_t n): _nw(n), _nx(n), _ny(n), _nz(n), _A(n*n*n*n) {}
FourTensor::FourTensor(size_t nw, size_t nx, size_t ny, size_t nz): _nw(nw), _nx(nx), _ny(ny), _nz(nz), _A(nw*nx*ny*nz) {}
FourTensor::FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, double *A): _nw(nw), _nx(nx), _ny(ny), _nz(nz), _A(A, A + nw*nx*ny*nz) {}
FourTensor::FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, Vector A): _nw(nw), _nx(nx), _ny(ny), _nz(nz), _A(nw*nx*ny*nz) {
	if(A.size() == _A.size()){
		_A = A;
	}else{
		std::cout << "ERROR: Sizes do not match \n";
	}
}
FourTensor::FourTensor(size_t nw, size_t nx, size_t ny, size_t nz, double val): _nw(nw), _nx(nx), _ny(ny), _nz(nz), _A(nw*nx*ny*nz, val) {}

size_t FourTensor::dim(size_t d) const{
	if (d == 0) return _nw;
	if (d == 1) return _nx;
	if (d == 2) return _ny;
	if (d == 3) return _nz;
	return -1;
}
size_t FourTensor::size() const{
	return _A.size();
}

ThreeTensor FourTensor::slice_w(size_t i){
	ThreeTensor slc(_nx, _ny, _nz);
	for(size_t j = 0; j < _nx; j++){
		for(size_t k = 0; k < _ny; k++){
			for(size_t l = 0; l < _nz; l++){
				slc(j, k, l) = _A[((i*_nx + j)*_ny + k)*_nz + l];
			}
		}
	}
	return slc;
}

ThreeTensor FourTensor::slice_x(size_t j){
	ThreeTensor slc(_nw, _ny, _nz);
	for(size_t i = 0; i < _nw; i++){
		for(size_t k = 0; k < _ny; k++){
			for(size_t l = 0; l < _nz; l++){
				slc(i, k, l) = _A[((i*_nx + j)*_ny + k)*_nz + l];
			}
		}
	}
	return slc;
}

ThreeTensor FourTensor::slice_y(size_t k){
	ThreeTensor slc(_nw, _nx, _nz);
	for(size_t i = 0; i < _nw; i++){
		for(size_t j = 0; j < _nx; j++){
			for(size_t l = 0; l < _nz; l++){
				slc(i, j, l) = _A[((i*_nx + j)*_ny + k)*_nz + l];
			}
		}
	}
	return slc;
}

ThreeTensor FourTensor::slice_z(size_t l){
	ThreeTensor slc(_nw, _nx, _ny);
	for(size_t i = 0; i < _nw; i++){
		for(size_t j = 0; j < _nx; j++){
			for(size_t k = 0; k < _ny; k++){
				slc(i, j, k) = _A[((i*_nx + j)*_ny + k)*_nz + l];
			}
		}
	}
	return slc;
}

ThreeTensor FourTensor::slice(size_t d, size_t ii){
	if(d == 0){
		return slice_w(ii);
	}else if(d == 1){
		return slice_x(ii);
	}else if(d == 2){
		return slice_y(ii);
	}else if(d == 3){
		return slice_z(ii);
	}else{
		std::cout << "ERROR: Invalid dimension \n";
		return ThreeTensor();
	}
}

Matrix FourTensor::slice_wx(size_t i, size_t j){
	Matrix slc(_ny, _nz);
	for(size_t k = 0; k < _ny; k++){
		for(size_t l = 0; l < _nz; l++){
			slc(k, l) = _A[((i*_nx + j)*_ny + k)*_nz + l];
		}
	}
	return slc;
}

Matrix FourTensor::slice_wy(size_t i, size_t k){
	Matrix slc(_nx, _nz);
	for(size_t j = 0; j < _nx; j++){
		for(size_t l = 0; l < _nz; l++){
			slc(j, l) = _A[((i*_nx + j)*_ny + k)*_nz + l];
		}
	}
	return slc;
}

Matrix FourTensor::slice_wz(size_t i, size_t l){
	Matrix slc(_nx, _ny);
	for(size_t j = 0; j < _nx; j++){
		for(size_t k = 0; k < _ny; k++){
			slc(j, k) = _A[((i*_nx + j)*_ny + k)*_nz + l];
		}
	}
	return slc;
}

Matrix FourTensor::slice_xy(size_t j, size_t k){
	Matrix slc(_nw, _nz);
	for(size_t i = 0; i < _nw; i++){
		for(size_t l = 0; l < _nz; l++){
			slc(i, l) = _A[((i*_nx + j)*_ny + k)*_nz + l];
		}
	}
	return slc;
}

Matrix FourTensor::slice_xz(size_t j, size_t l){
	Matrix slc(_nw, _ny);
	for(size_t i = 0; i < _nw; i++){
		for(size_t k = 0; k < _ny; k++){
			slc(i, k) = _A[((i*_nx + j)*_ny + k)*_nz + l];
		}
	}
	return slc;
}

Matrix FourTensor::slice_yz(size_t k, size_t l){
	Matrix slc(_nw, _nx);
	for(size_t i = 0; i < _nw; i++){
		for(size_t j = 0; j < _nx; j++){
			slc(i, j) = _A[((i*_nx + j)*_ny + k)*_nz + l];
		}
	}
	return slc;
}

Matrix FourTensor::slice(size_t d1, size_t d2, size_t i1, size_t i2){
	if(d1 == 0 and d2 == 1){
		return slice_wx(i1, i2);
	}else if(d1 == 0 and d2 == 2){
		return slice_wy(i1, i2);
	}else if(d1 == 0 and d2 == 3){
		return slice_wz(i1, i2);
	}else if(d1 == 1 and d2 == 2){
		return slice_xy(i1, i2);
	}else if(d1 == 1 and d2 == 3){
		return slice_xz(i1, i2);
	}else if(d1 == 2 and d2 == 3){
		return slice_yz(i1, i2);
	}else{
		std::cout << "ERROR: Invalid dimension \n";
		return Matrix();
	}
}

Vector FourTensor::slice_wxy(size_t i, size_t j, size_t k){
	Vector slc(_nz);
	for(size_t l = 0; l < _nz; l++){
		slc[l] = _A[((i*_nx + j)*_ny + k)*_nz + l];
	}
	return slc;
}

Vector FourTensor::slice_wxz(size_t i, size_t j, size_t l){
	Vector slc(_ny);
	for(size_t k = 0; k < _ny; k++){
		slc[k] = _A[((i*_nx + j)*_ny + k)*_nz + l];
	}
	return slc;
}

Vector FourTensor::slice_wyz(size_t i, size_t k, size_t l){
	Vector slc(_nx);
	for(size_t j = 0; j < _nx; j++){
		slc[j] = _A[((i*_nx + j)*_ny + k)*_nz + l];
	}
	return slc;
}

Vector FourTensor::slice_xyz(size_t j, size_t k, size_t l){
	Vector slc(_nw);
	for(size_t i = 0; i < _nw; i++){
		slc[i] = _A[((i*_nx + j)*_ny + k)*_nz + l];
	}
	return slc;
}

Vector FourTensor::slice(size_t d1, size_t d2, size_t d3, size_t i1, size_t i2, size_t i3){
	if(d1 == 0 and d2 == 1 and d3 == 2){
		return slice_wxy(i1, i2, i3);
	}else if(d1 == 0 and d2 == 1 and d3 == 3){
		return slice_wxz(i1, i2, i3);
	}else if(d1 == 0 and d2 == 2 and d3 == 3){
		return slice_wyz(i1, i2, i3);
	}else if(d1 == 1 and d2 == 2 and d3 == 3){
		return slice_xyz(i1, i2, i3);
	}else{
		std::cout << "ERROR: Invalid dimensions \n";
		return Vector();
	}
}

void FourTensor::reshape(size_t nw, size_t nx, size_t ny, size_t nz){
	_nw = nw;
	_nx = nx;
	_ny = ny;
	_nz = nz;
}

FourTensor FourTensor::reshaped(size_t nw, size_t nx, size_t ny, size_t nz) const{
	return FourTensor(nw, nx, ny, nz, _A);
}

void FourTensor::set_value(size_t i, size_t j, size_t k, size_t l, double val){
	_A[((i*_nx + j)*_ny + k)*_nz + l] = val;
}

double& FourTensor::operator()(size_t i, size_t j, size_t k, size_t l){
	return _A[((i*_nx + j)*_ny + k)*_nz + l];
}
const double& FourTensor::operator()(size_t i, size_t j, size_t k, size_t l) const{
	return _A[((i*_nx + j)*_ny + k)*_nz + l];
}

Vector FourTensor::data(){
	return _A;
}

//////////////////////////////////////////////////////////////////
//////////////            StopWatch Class         ////////////////
//////////////////////////////////////////////////////////////////

StopWatch::StopWatch():time_elapsed(0.), t1(std::chrono::high_resolution_clock::now()), t2(t1) {}
void StopWatch::start(){
	t1 = std::chrono::high_resolution_clock::now();
	t2 = t1;
}

void StopWatch::stop(){
	t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	time_elapsed += time_span.count();
}

void StopWatch::reset(){
	time_elapsed = 0.;
}

void StopWatch::print(){
	std::cout << "It took me "<< time_elapsed <<" seconds total.";
	std::cout << std::endl;
}

void StopWatch::print(size_t cycles){
	std::cout << "It took me "<< time_elapsed/cycles <<" seconds per cycle, "<< time_elapsed <<" seconds total.";
	std::cout << std::endl;
}

double StopWatch::time(){
	return time_elapsed;
}
