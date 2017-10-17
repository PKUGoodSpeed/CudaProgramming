#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "matrix.hpp"

const int rA = 1024;
const int cA = 1024;
const int rB = 1024;
const int cB = 1024;

using namespace std;

int main(){
	vector<float> A(rA*cA), B(rB*cB);
	srand(0);
	generate(A.begin(), A.end(), [](){return (float)rand()/RAND_MAX;});
	generate(B.begin(), B.end(), [](){return (float)rand()/RAND_MAX;});
	cout << setprecision(6);
	vector<float> C = MatrixMultiplication<cpu, float>()(A, B, rA, cA, rB, cB);
	for(int i=0;i<rA;i+=rA/6+1){
		for(int j=0;j<cB;j+=cB/6+1) cout<<C[i*cB + j]<<' ';
		cout<<endl;
	}
	C = MatrixMultiplication<gpu, float>()(A, B, rA, cA, rB, cB);
	for(int i=0;i<rA;i+=rA/6+1){
		for(int j=0;j<cB;j+=cB/6+1) cout<<C[i*cB + j]<<' ';
		cout<<endl;
	}
	C = MatrixMultiplication<cublas, float>()(A, B, rA, cA, rB, cB);
	for(int i=0;i<rA;i+=rA/6+1){
		for(int j=0;j<cB;j+=cB/6+1) cout<<C[i*cB + j]<<' ';
		cout<<endl;
	}
	return 0;
}
