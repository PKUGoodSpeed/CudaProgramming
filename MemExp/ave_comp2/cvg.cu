#include <bits/stdc++.h>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>
using namespace std;

const int ARRAY_SIZE = 1E9;

__global__ void initKernel(){
    return;
}

__global__ void naiveKernel(int N, float *input, float *output){
    float res = 0.;
    for(int i=0;i<N;++i) res += input[i];
    *output = res/N;
}

__global__ void thrustKernel(int N, float *input, float *output){
    float res = thrust::reduce(thrust::device, input, input + N);
    *output = res/N;
}

int main(){
    cudaEvent_t start, stop;
    float cpu_time, gpu_time1, gpu_time2;
    cudaEventCreate(&start);   // creating the event 1
    cudaEventCreate(&stop);    // creating the event 2
    initKernel<<<1,1>>>();
    initKernel<<<1,1>>>();
    initKernel<<<1,1>>>();
    for(int N = 2; N<=ARRAY_SIZE ; N*=2){
        float ans = 0.;
        vector<float> input(N);
        def_dvec(float) dev_in(N), dev_ans1(1, 0.), dev_ans2(1,0.);
        generate(input.begin(), input.end(), [](){return float(rand())/RAND_MAX;});
        gpu_copy(input, dev_in);
        
        // Using CPU to compute the average
        clock_t t_start = clock();
        ans = accumulate(input.begin(), input.end(), 0.)/N;
        cpu_time = float(clock() - t_start)/CLOCKS_PER_SEC;
        
        // Using the naive kernel
        cudaEventRecord(start, 0);                 
        naiveKernel<<<1, 1>>>(N, to_ptr(dev_in), to_ptr(dev_ans1));
        cudaEventRecord(stop, 0);                  // Stop time measuring
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time1, start, stop);
        gpu_time1/=1000.;
        
        // Using the thrust kernel
        cudaEventRecord(start, 0);                 
        thrustKernel<<<1, 1>>>(N, to_ptr(dev_in), to_ptr(dev_ans2));
        cudaEventRecord(stop, 0);                  // Stop time measuring
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_time2, start, stop);
        gpu_time2 /= 1000.;
        
        // output results
        cout<< '[' << N<<','<<cpu_time<<','<<gpu_time1<<','<<gpu_time2<<',';
        cout<< ans <<','<<dev_ans1[0]<<','<<dev_ans2[0]<<"],"<<endl;
    }
    
    return 0;
}
