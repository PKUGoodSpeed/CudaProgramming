#include <iostream>
#include <cassert>
#include <ctime>
#include <thrust/device_vector.h>
#include <thurst/inner_product.h>
#include <./matrix.hpp>

#define cu_dot2(x, y, n) thrust::inner_product((x), (x) + (n), (y), 0)
using namespace std;

template<>
vector<int> MatrixMultiplication<gpu, int>::operator ()(const vector<int> &A, const vector<int> &B, int rA, int cA, int rB, int cB){
	assert((int)A.size() == rA * cA);
	assert((int)B.size() == rB * cB);
	assert(cA == rB);
	clock_t t_start = clock(), t_end;
	clock_t t_end = clock();
	cout<<"GPU Matrix Multiplication Time Usage:"<<endl;
	cout<< double(t_end - t_start)/CLOCKS_PER_SEC << " s"<<endl;
	return C;
}
