#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
//#include "inner.hpp"

#define t_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define t_tran_u(x, z, u) thrust::transform((x).begin(), (x).end(), (z).begin(), u)
#define t_tran_b(x, y, z, b) thrust::transform((x).begin(), (x).end(), (y).begin(), (z).begin(), b)
#define t_sum(x) thrust::reduce((x).begin(), (x).end(), 0)

using namespace std;

typedef thrust::device_vector<int> dvi;

int innerSerial(const vector<int> &x, const vector<int> &y){
	assert(x.size() == y.size());
	int n = (int)x.size();
	int res = 0;
	for(int i=0;i<n;++i) res += x[i]*y[i];
	return res;
}

int innerParallel(const vector<int> &x, const vector<int> &y){
	assert(x.size() == y.size());
	dvi dx = x, dy = y;
	t_tran_b(dx, dy, dx, thrust::multiplies<int>());
	return t_sum(dx);
}

int innerProduct(const vector<int> &x, const vector<int> &y, bool parallel){
	if(parallel) return innerParallel(x, y);
	return innerSerial(x, y);
}


