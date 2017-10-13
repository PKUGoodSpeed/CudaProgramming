#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include "inner.hpp"

#define cu_dot(x, y) thrust::inner_product((x).begin(), (x).end(), (y).begin(), 0)
using namespace std;

typedef thrust::device_vector<int> dvi;

template<>
int InnerProd<gpu, int>::operator ()(const vector<int> &x, const vector<int> &y){
	dvi dx = x, dy = y;
	return cu_dot(dx, dy);
}


