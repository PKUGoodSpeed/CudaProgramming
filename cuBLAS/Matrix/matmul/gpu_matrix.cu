#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <cassert>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include "./matrix.hpp"

using namespace std;
typedef thrust::device_vector<float> dvf;

const int BLOCK_SIZE = 32;
const int NUM_BLOCKS = 512;

__global__ void matmulKernel(float *A, float *B, float *C, int rA, int cA, int cB){
	int i = blockIdx.y*gridDim.x + blockIdx.x, j = threadIdx.y*blockDim.x + threadIdx.x;
	if(i < rA && j < cB){
		C[i*cB + j] = 0.;
		for(int k=0;k<cA;++k) C[i*cB + j] += A[i*cA + k] * B[k*cB + j];
	}
	return;
}

template<>
vector<float> MatrixMultiplication<gpu, float>::operator ()(const vector<float> &A, const vector<float> &B, int rA, int cA, int rB, int cB){
	assert((int)A.size() == rA * cA);
	assert((int)B.size() == rB * cB);
	assert(cA == rB);
	vector<float> C(rA * cB);
	float *dA, *dB, *dC;
	clock_t t_start = clock(), t_end;
	cudaMalloc((void **)&dA, rA*cA*sizeof(float));
	cudaMalloc((void **)&dB, rB*cB*sizeof(float));
	cudaMalloc((void **)&dC, rA*cB*sizeof(float));
	// Compute matrix multiplication:
	cudaMemcpy(dA, &A[0], rA*cA*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, &B[0], rB*cB*sizeof(float), cudaMemcpyHostToDevice);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE), dimGrid(NUM_BLOCKS, NUM_BLOCKS);
	matmulKernel<<<dimGrid, dimBlock>>>(dA, dB, dC, rA, cA, cB);
	cudaMemcpy(&C[0], dC, rA*cB*sizeof(float), cudaMemcpyDeviceToHost);
	t_end = clock();
	cout<<"GPU Matrix Multiplication Time Usage:"<<endl;
	cout<< double(t_end - t_start)/CLOCKS_PER_SEC << " s"<<endl;
	cout<<endl;
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
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
	if (status != CUBLAS_STATUS_SUCCESS) cerr << "Kernel execution error!\n";
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS) cerr << "!!!! shutdown error (A)\n";
	vector<float> C(rA * cB);
	thrust::copy(dC.begin(), dC.end(), C.begin());
	t_end = clock();
	cout<<"CUBLAS Matrix Multiplication Time Usage:"<<endl;
	cout<< double(t_end - t_start)/CLOCKS_PER_SEC << " s"<<endl;
	cout<<endl;
	return C;
}

/*
int main(){
	vector<float> A(1024*1024, 2.), B(1024*1024, 1.);
	int rA = 1024, cB = 1024, cA = 1024, rB = 1024;
	vector<float> C = MatrixMultiplication<gpu, float>()(A, B, rA, cA, rB, cB);
	cout<<"C:"<<endl;
	for(int i=0;i<rA;i+=rA/10+1){
		for(int j=0;j<cB;j+=cB/10+1) cout<<C[i*cB + j]<<' ';
		cout<<endl;
	}
	return 0;
}*/
