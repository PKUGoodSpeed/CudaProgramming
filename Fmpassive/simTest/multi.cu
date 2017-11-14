#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int BLOCK_SIZE = 1024;
const int VEC_SIZE = 512;

__global__ void init(){}

__global__ void process(int N_step, int N_inst, float *input, float *output){
    int b_id = blockIdx.x, t_id = threadIdx.x;
    if(b_id >= N_inst) return;
    __shared__ float ans;
    float val;
    if(!t_id) ans = 0;
    if(t_id < VEC_SIZE) val = input[VEC_SIZE * b_id + t_id];
    __syncthreads();
    for(int t=0;t<N_step;++t){
        int start = t%VEC_SIZE;
        if(t_id >= start && t_id < min(start + 12, VEC_SIZE)) atomicAdd(&ans, val);
        if(start + 12 > VEC_SIZE && t_id < start + 12 - VEC_SIZE) atomicAdd(&ans, val);
        __syncthreads();
    }
    output[b_id] = ans;
    return;
}

int main(int argc, char *argv[]){
    srand(0);
    int num_inst = 1024, num_step = 1024;
    if(argc > 1) num_step = stoi(argv[1]);
    /* For measuing the time */
    cudaEvent_t start, stop;
    float cuda_time;
    cudaEventCreate(&start);   // creating the event 1
    cudaEventCreate(&stop);    // creating the event 2
    vector<float> hin(VEC_SIZE * num_step), hout(num_inst);
    def_dvec(float) din(VEC_SIZE * num_step), dout(num_inst);
    generate(hin.begin(), hin.end(), [](){return float(rand())/RAND_MAX;});
    int n_block = num_inst;
    init<<<n_block,BLOCK_SIZE>>>();
    gpu_copy(hin, din);
    cudaEventRecord(start, 0);
    process<<<n_block, BLOCK_SIZE>>>(num_step, num_inst, to_ptr(din), to_ptr(dout));
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for running the kernel is: "<<cuda_time/1000<<"s"<<endl;
    gpu_copy(dout, hout);
    cout<<"Showing the answer:"<<endl;
    for(int i=0;i<num_inst;i+=num_inst/10) cout<<hout[i]<<' ';
    cout<<endl;
    return 0;
}

