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
	return vector<int>();
}

int main(){
	vector<int> A = {1,2,3,4,5,6}, B = {1,2,3,4,5,6};
	int rA = 2, cB = 2, cA = 3, rB = 3;
	cout<<"A:"<<endl;
	for(int i=0;i<rA;++i){
		for(int j=0;j<cA;++j) cout<<A[i*cA + j]<<' ';
		cout<<endl;
	}
	cout<<"B:"<<endl;
	for(int i=0;i<rB;++i){
		for(int j=0;j<cB;++j) cout<<B[i*cB + j]<<' ';
		cout<<endl;
	}
	vector<int> C = MatrixMultiplication<cpu, int>()(A, B, rA, cA, rB, cB);
	cout<<"C:"<<endl;
	for(int i=0;i<rA;++i){
		for(int j=0;j<cB;++j) cout<<C[i*cB + j]<<' ';
		cout<<endl;
	}
	return 0;
}