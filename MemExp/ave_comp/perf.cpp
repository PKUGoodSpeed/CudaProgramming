#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
using namespace std;

const int BLOCK_SIZE = 1024;
const int SHARE_SIZE = 1024;

__global__ void naiveKernel(int N, float *input, float *output){
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_i < N){
        for(int i=0;i<N;++i) output[global_i] += input[i];
        output[global_i] /= N;
    }
    return ;
}

__global__ void smemKernel(int N, float *input, float *output){
    int b_size = blockDim.x, b_idx = blockIdx.x, t_idx = threadIdx.x;
    int global_i = b_size * b_idx + t_idx, n_chk = (N + SHARE_SIZE - 1)/SHARE_SIZE;
    __shared__ float buff[SHARE_SIZE];
    for(int q=0;q<n_chk;++q){
        int left = q*SHARE_SIZE, right = min(left + SHARE_SIZE, N);
        for(int i = t_idx + left; i < right; i += b_size) buff[i-left] = input[i];
        __syncthreads();
        if(global_i < N){
            for(int i = left; i < right; ++i) output[global_i] += buff[i-left];
        }
        __syncthreads();
    }
    output[global_i] /= N;
    return ;
}

int main(int argc, char *argv[]){
    for(int N = 16; N < (1<<25); N*=2){
    float *input = new float [N], *output = new float [N];
    float *dev_in, *dev_out;
    clock_t time;
    cudaMalloc((void **)&dev_in, N*sizeof(float));
    cudaMalloc((void **)&dev_out, N*sizeof(float));
    for(int i=0;i<N;++i) input[i] = (float)rand()/RAND_MAX;
    
    /* Using serial code */
    time = clock();
    float ans = accumulate(input, input + N, 0.)/N;
    cout << '[' << N << ',' <<float(clock() - time)/CLOCKS_PER_SEC << ',';
    
    /* Doing parallel */
    int block_size = BLOCK_SIZE;
    int num_block = (N + block_size - 1)/block_size;
    
    cudaEvent_t start, stop;
    float cuda_time;
    cudaEventCreate(&start);   // creating the event 1
    cudaEventCreate(&stop);    // creating the event 2
    
    for(int q = 0; q < 6; ++q){
    cudaMemcpy(dev_in , input, N*sizeof(float), cudaMemcpyHostToDevice);
    memset(output, 0, N*sizeof(float));
    cudaMemcpy(dev_out, output, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);                 // Start time measuring
    smemKernel<<<num_block, block_size>>>(N, dev_in, dev_out);
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    if(q==4) cout<< cuda_time/1000 <<"],"<<endl;
    cudaMemcpy(output, dev_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(dev_in);
    cudaFree(dev_out);
    delete [] input;
    delete [] output;
    }
    
    
    
    return 0;
}