#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;

const int BLOCK_SIZE = 64;
const SHARE_SIZE = 1000;

__global__ naiveKernel(int N, double *input, double *output){
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_i < N){
        output[global_i] = 0.;
        for(int i=0;i<N;++i) output[global_i] += input[i];
        output[global_i] /= N;
    }
}

__global__ smemKernel(int N, double *input, double *output){
    int b_size = blockDim.x, b_idx = blockIdx.x, t_idx = threadIdx.x;
    int global_i = b_size * b_idx + t_idx, n_chk = (N + SHARE_SIZE - 1)/SHARE_SIZE;
    __shared__ float buff[SHARE_SIZE];
    for(int q=0;q<n_chk;++q){
        int left = q*SHARE_SIZE, right = min(left + SHARE_SIZE, N);
        for(int i = t_idx + left; i < right; i += b_size);
        __syncthreads();
    }
}
