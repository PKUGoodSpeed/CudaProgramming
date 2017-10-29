#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;

const int BLOCK_SIZE = 64;
const SHARE_SIZE = 16;

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
        for(int i = t_idx + left; i < right; i += b_size) buff[i-left] = input[i];
        __syncthreads();
        for(int i = left; i < right; ++i) output[global_i] += buff[i-left];
        __syncthreads();
    }
    output[global_i] /= N;
}

int main(int argc, char *argv[]){
    N = 1<<24;
    vector<double> input(N);
    clock_t time;
    generate(input.begin(), input.end(), [](){return double(rand())/RAND_MAX;});
    
    /* Using serial code */
    time = clock();
    cout << "Serial (CPU) Code:" << endl;
    cout << "Time Usage: " << double(clock() - time)/CLOCKS_PER_SEC << endl;
    cout << "Answer: " << accumulate(input.begin(), input.end(), 0.)/N << endl << endl;
    
    
    /* Doing parallel */
    int block_size = BLOCK_SIZE;
    if(argc > 1) block_size = stoi(string(argv[1]));
    int num_block = (N + block_size - 1)/block_size;
    cout << "block_size = " << block_size << endl;
    cout << "num_blocks = " << num_block << endl << endl;
    
    /* First, without using shared memory */
    time = clock();
    thrust::device_vector<double> dev_in = input, dev_out(N, 0.);
    naiveKernel<<<num_block, block_size>>>(N, to_ptr(dev_in), to_ptr(dev_out));
    cout << "GPU code without using shared memory: " << endl;
    cout << "Time Usage: " << double(clock() - time)/CLOCKS_PER_SEC << endl;
    cout << "Answer: " << endl;
    for(int i=0;i<N;i+=N/12+1) cout << dev_out[i] << ' ';
    cout<< endl << endl;
    
    /* Second, using shared memory */
    time = clock();
    thrust::device_vector<double> dev_in = input, dev_out(N, 0.);
    naiveKernel<<<num_block, block_size>>>(N, to_ptr(dev_in), to_ptr(dev_out));
    cout << "GPU code without using shared memory: " << endl;
    cout << "Time Usage: " << double(clock() - time)/CLOCKS_PER_SEC << endl;
    cout << "Answer: " << endl;
    for(int i=0;i<N;i+=N/12+1) cout << dev_out[i] << ' ';
    cout<< endl << endl;
    
}
