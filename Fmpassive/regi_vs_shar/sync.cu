#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int VEC_SIZE = 12288;

__global__ void init(){}

__device__ float doThings(int t,int sz, float *input){
    float ans = 0;
    for(int i=0;i<10;++i){
        ans += input[(i+t)%sz];
    }
    return ans;
}

__global__ void process(int N_step, float *input){
    __shared__ float local_data[VEC_SIZE];
    thrust::copy(thrust::device, input, input + VEC_SIZE, local_data);
    for(int t=0;t<N_step;++t){
        doThings(t, VEC_SIZE, local_data);
    }
}

__global__ void syncprocess(int N_step, float *input){
    __shared__ float local_data[VEC_SIZE];
    thrust::copy(thrust::device, input, input + VEC_SIZE, local_data);
    for(int t=0;t<N_step;++t){
        doThings(t, VEC_SIZE, local_data);
        __syncthreads();
    }
}

int main(int argc, char *argv[]){
    srand(0);
    int num_step = 1024*1024;
    if(argc > 1) num_step = stoi(argv[1]);
    /* For measuing the time */
    cudaEvent_t start, stop;
    float cuda_time;
    cudaEventCreate(&start);   // creating the event 1
    cudaEventCreate(&stop);    // creating the event 2
    vector<float> hin(VEC_SIZE);
    def_dvec(float) din(VEC_SIZE);
    generate(hin.begin(), hin.end(), [](){return float(rand())/RAND_MAX;});
    init<<<1,1024>>>();
    gpu_copy(hin, din);
    cudaEventRecord(start, 0);
    process<<<1, 1024>>>(num_step, to_ptr(din));
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for running the asynchronized kernel is: "<<cuda_time/1000<<"s"<<endl<<endl;

    cudaEventRecord(start, 0);
    syncprocess<<<1, 1024>>>(num_step, to_ptr(din));
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for running the synchronized kernel is: "<<cuda_time/1000<<"s"<<endl<<endl;
    return 0;
}