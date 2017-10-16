#include <iostream>
#include <cassert>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include "./matrix.hpp"

#define device_dot(x, y, n) thrust::inner_product(thrust::device, (x), (x) + (n), (y), 0)
using namespace std;

const int BLOCK_SIZE = 512;
const int NUM_BLOCKS = 1024;

__global__ void matmulKernel(int *A, int *B, int *BT, int *C, int rA, int cA, int cB, int *sync){
	int i = blockIdx.y*gridDim.x + blockIdx.x, j = threadIdx.y*blockDim.x + threadIdx.x;
	if(i < cA && j < cB) BT[j*cA + i] = B[i*cB + j];
	atomicAdd(sync, 0);
	if(i < rA && j < cB){
		C[i*cB + j] = device_dot(&A[i*cA], &BT[j*cA], cA);
	}
	return;
}

template<>
vector<int> MatrixMultiplication<gpu, int>::operator ()(const vector<int> &A, const vector<int> &B, int rA, int cA, int rB, int cB){
	assert((int)A.size() == rA * cA);
	assert((int)B.size() == rB * cB);
	assert(cA == rB);
	vector<int> C(rA * cB);
	int *dA, *dB, *dC, *tB, *sync;
	clock_t t_start = clock(), t_end;
	cudaMalloc((void **)&dA, rA*cA*sizeof(int));
	cudaMalloc((void **)&dB, rB*cB*sizeof(int));
	cudaMalloc((void **)&tB, rB*cB*sizeof(int));   // We need to compute the transpose of B
	cudaMalloc((void **)&dC, rA*cB*sizeof(int));
	cudaMalloc((void **)&sync, 1*sizeof(int));

	cudaMemcpy(dA, &A[0], rA*cA*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, &B[0], rB*cB*sizeof(int), cudaMemcpyHostToDevice);
	matmulKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(dA, dB, tB, dC, rA, cA, cB, sync);
	cudaMemcpy(&C[0], dC, rA*cB*sizeof(int), cudaMemcpyDeviceToHost);
	t_end = clock();
	cout<<"GPU Matrix Multiplication Time Usage:"<<endl;
	cout<< double(t_end - t_start)/CLOCKS_PER_SEC << " s"<<endl;
	cout<<endl;
	return C;
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
	vector<int> C = MatrixMultiplication<gpu, int>()(A, B, rA, cA, rB, cB);
	cout<<"C:"<<endl;
	for(int i=0;i<rA;++i){
		for(int j=0;j<cB;++j) cout<<C[i*cB + j]<<' ';
		cout<<endl;
	}
	return 0;
}
