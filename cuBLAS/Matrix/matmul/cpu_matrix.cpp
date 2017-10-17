#include <iostream>
#include <cassert>
#include <ctime>
#include "./matrix.hpp"
using namespace std;

template<>
vector<float> MatrixMultiplication<cpu, float>::operator ()(const vector<float> &A, const vector<float> &B, int rA, int cA, int rB, int cB){
	clock_t t_start = clock(), t_end;
	assert((int)A.size() == rA * cA);
	assert((int)B.size() == rB * cB);
	assert(cA == rB);
	vector<float> C(rA * cB, 0.);
	for(int i=0;i<rA;++i) for(int j=0;j<cB;++j) for(int k=0;k<cA;++k) C[i*cB + j] += A[i*cA + k] * B[k*cB + j];
	t_end = clock();
	cout<<"CPU Matrix Multiplication Time Usage:\n";
	cout<<"\t"<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl;
	return C;
}
