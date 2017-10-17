#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <cassert>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <cublas_v2.h>
#include "./matrix.hpp"

#define device_dot(x, y, n) thrust::inner_product(thrust::device, (x), (x) + (n), (y), 0)
using namespace std;
typedef thrust::device_vector<float> dvf;

const int BLOCK_SIZE = 512;
const int NUM_BLOCKS = 1024;

__global__ void matmulKernel(float *A, float *B, float *BT, float *C, int rA, int cA, int cB, int *sync){
	int i = blockIdx.y*gridDim.x + blockIdx.x, j = threadIdx.y*blockDim.x + threadIdx.x;
	if(i < cA && j < cB) BT[j*cA + i] = B[i*cB + j];
	atomicAdd(sync, 0);
	if(i < rA && j < cB){
		C[i*cB + j] = device_dot(&A[i*cA], &BT[j*cA], cA);
	}
	return;
}

template<>
vector<float> MatrixMultiplication<gpu, float>::operator ()(const vector<float> &A, const vector<float> &B, int rA, int cA, int rB, int cB){
	assert((int)A.size() == rA * cA);
	assert((int)B.size() == rB * cB);
	assert(cA == rB);
	vector<float> C(rA * cB);
	float *dA, *dB, *dC, *tB;
	int *sync;
	clock_t t_start = clock(), t_end;
	cudaMalloc((void **)&dA, rA*cA*sizeof(float));
	cudaMalloc((void **)&dB, rB*cB*sizeof(float));
	cudaMalloc((void **)&tB, rB*cB*sizeof(float));   // We need to compute the transpose of B
	cudaMalloc((void **)&dC, rA*cB*sizeof(float));
	cudaMalloc((void **)&sync, 1*sizeof(int));

	cudaMemcpy(dA, &A[0], rA*cA*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, &B[0], rB*cB*sizeof(float), cudaMemcpyHostToDevice);
	matmulKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(dA, dB, tB, dC, rA, cA, cB, sync);
	cudaMemcpy(&C[0], dC, rA*cB*sizeof(float), cudaMemcpyDeviceToHost);
	t_end = clock();
	cout<<"GPU Matrix Multiplication Time Usage:"<<endl;
	cout<< double(t_end - t_start)/CLOCKS_PER_SEC << " s"<<endl;
	cout<<endl;
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(tB);
	cudaFree(dC);
	cudaFree(sync);
	return C;
}

template<>
vector<float> MatrixMultiplication<cublas, float>::operator ()(const vector<float> &A, const vector<float> &B, int rA, int cA, int rB, int cB){
	assert((int)A.size() == rA * cA);
	assert((int)B.size() == rB * cB);
	assert(cA == rB);
	dvf dC(rA * cB), dA = A, dB = B;
	cublasHandle_t handle;
	clock_t t_start = clock(), t_end;
	
	/* Initialization of cuBLAS */
	cublasStatus_t status = cublasCreate(&handle);
  	if(status != CUBLAS_STATUS_SUCCESS) cerr << "CUBLAS initialization error!\n";
  
  	float alpha = 1.0f, beta = 0.0f;
  	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                      cB, rA, cA, 
                                      &alpha, thrust::raw_pointer_cast(&dB[0]), cB, 
                                              thrust::raw_pointer_cast(&dA[0]), cA, 
                                      &beta,  thrust::raw_pointer_cast(&dC[0]), cB);
	t_end = clock();
	cout<<"CUBLAS Matrix Multiplication Time Usage:"<<endl;
	cout<< double(t_end - t_start)/CLOCKS_PER_SEC << " s"<<endl;
	cout<<endl;
  	if (status != CUBLAS_STATUS_SUCCESS) cerr << "Kernel execution error!\n";
	status = cublasDestroy(handle);
  	if (status != CUBLAS_STATUS_SUCCESS) cerr << "!!!! shutdown error (A)\n";

	return vector<float>(dC.begin(), dC.end());
}

int main(){
	vector<float> A(256*256, 1.), B(256*256, 1.);
	int rA = 256, cB = 256, cA = 256, rB = 256;
	vector<float> C = MatrixMultiplication<cublas, float>()(A, B, rA, cA, rB, cB);
	cout<<"C:"<<endl;
	for(int i=0;i<rA;i+=rA/10+1){
		for(int j=0;j<cB;j+=cB/10+1) cout<<C[i*cB + j]<<' ';
		cout<<endl;
	}
	return 0;
}
