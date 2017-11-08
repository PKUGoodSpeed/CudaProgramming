#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;
const int BLOCK_SIZE = 512;

__global__ void init(){}

__global__ void scatterSum(int N, float *input, float *output){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    for(int j=0;j<N;++j){
        atomicAdd(output+j, input[i]);
    }
    return;
}

int main(int argc, char* argv[]){
    int N = 1024*1024;
    if(argc > 1) N = stoi(argv[1]);
    
    cudaEvent_t start, stop;
    float cuda_time;
    cudaEventCreate(&start);   // creating the event 1
    cudaEventCreate(&stop);    // creating the event 2
    
    cudaEventRecord(start, 0);
    def_dvec(float) input(N, 1.), output(N, 0.);
    int num_blocks = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    init<<<num_blocks, BLOCK_SIZE>>>();
    scatterSum<<<num_blocks, BLOCK_SIZE>>>(N, to_ptr(input), to_ptr(output));
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for input oriented parallelism is: "<<cuda_time/1000<<"s"<<endl;
    for(int i=0;i<N;i+=N/10) cout<<output[i]<<' ';
    cout<<endl;
    return 0;
}