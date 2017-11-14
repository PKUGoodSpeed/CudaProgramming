#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int BLOCK_SIZE = 256;
const int VEC_SIZE = 512;

__global__ void init(){}

__global__ void process(int N_step, int N_inst, float *input, float *output){
    int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(g_id >= N_inst) return;
    float ans = 0.;
    for(int t=0;t<N_step;++t){
        for(int i=0;i<12;++i){
            ans += input[(i+t)%VEC_SIZE + VEC_SIZE * g_id];
        }
    }
    output[g_id] = ans;
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
    int n_block = (num_inst + BLOCK_SIZE - 1)/BLOCK_SIZE;
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