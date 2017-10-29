#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
using namespace std;

const int BLOCK_SIZE = 1024;
const int SHARE_SIZE = 16;

__global__ void naiveKernel(int N, double *input, double *output){
    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_i < N){
        output[global_i] = 0.;
        for(int i=0;i<N;++i) output[global_i] += input[i];
        output[global_i] /= N;
    }
    return ;
}

__global__ void smemKernel(int N, double *input, double *output){
    int b_size = blockDim.x, b_idx = blockIdx.x, t_idx = threadIdx.x;
    int global_i = b_size * b_idx + t_idx, n_chk = (N + SHARE_SIZE - 1)/SHARE_SIZE;
    __shared__ double buff[SHARE_SIZE];
    for(int q=0;q<n_chk;++q){
        int left = q*SHARE_SIZE, right = min(left + SHARE_SIZE, N);
        for(int i = t_idx + left; i < right; i += b_size) buff[i-left] = input[i];
        __syncthreads();
        for(int i = left; i < right; ++i) output[global_i] += buff[i-left];
        __syncthreads();
    }
    output[global_i] /= N;
    return ;
}

int main(int argc, char *argv[]){
    int N = 1<<24;
    double *input = new double [N], *output = new double [N];
    double *dev_in, *dev_out;
    clock_t time;
    cudaMalloc((void **)&dev_in, N*sizeof(double));
    cudaMalloc((void **)&dev_out, N*sizeof(double));
    for(int i=0;i<N;++i) input[i] = (double)rand()/RAND_MAX;
    cudaMemcpy(dev_in , input, N*sizeof(double), cudaMemcpyHostToDevice);
    
    /* Using serial code */
    time = clock();
    cout << "Serial (CPU) Code:" << endl;
    double ans = accumulate(input, input + N, 0.)/N;
    cout << "Time Usage: " << double(clock() - time)/CLOCKS_PER_SEC << endl;
    cout << "Answer: " << ans << endl << endl;
    
    /* Doing parallel */
    int block_size = BLOCK_SIZE;
    if(argc > 1) block_size = stoi(string(argv[1]));
    int num_block = (N + block_size - 1)/block_size;
    cout << "block_size = " << block_size << endl;
    cout << "num_blocks = " << num_block << endl << endl;
    
    /* First, without using shared memory */
    memset(output, 0, sizeof(output));
    time = clock();
    cudaMemcpy(dev_out, output, N*sizeof(double), cudaMemcpyHostToDevice);
    naiveKernel<<<num_block, block_size>>>(N, dev_in, dev_out);
    cout << "GPU code without using shared memory: " << endl;
    cout << "Time Usage: " << double(clock() - time)/CLOCKS_PER_SEC << endl;
    cudaMemcpy(output, dev_out, N*sizeof(double), cudaMemcpyDeviceToHost);
    cout << "Answer: " << endl;
    for(int i=0;i<N; i+=N/12+1) cout << output[i] << ' ';
    cout<< endl << endl;
    
    /* Second, using shared memory */
    memset(output, 0, sizeof(output));
    time = clock();
    cudaMemcpy(dev_out, output, N*sizeof(double), cudaMemcpyHostToDevice);
    smemKernel<<<num_block, block_size>>>(N, dev_in, dev_out);
    cout << "GPU code without using shared memory: " << endl;
    cout << "Time Usage: " << double(clock() - time)/CLOCKS_PER_SEC << endl;
    cudaMemcpy(output, dev_out, N*sizeof(double), cudaMemcpyDeviceToHost);
    cout << "Answer: " << endl;
    for(int i=0;i<N; i+=N/12+1) cout << output[i] << ' ';
    cout<< endl << endl;
    
}
