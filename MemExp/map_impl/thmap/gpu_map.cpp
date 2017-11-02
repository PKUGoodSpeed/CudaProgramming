/*
 We consider 1024 map, which are initially constructed with (at most)size = 1024 key, value pairs, for which keys are integers and values are float point numbers.
 Then each of them processes 262,144 operations. 
 The operations include:
 'i': insert a key-value pair, or modify the original value. (if the map is full, do nothing)
 'r': remove a key.
 'g': get the value for a key. If that key dose not exist, return 0
 */
 
#include <bits/stdc++.h>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define gpu_seq(x) thrust::sequence((x).begin(), (x).end())
#define host_find(x, n, key) thrust::find((x), (x)+n, key);
#define device_find(x, n, key) thrust::find(thrust::device, (x), (x)+n, key)
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int BLOCK_SIZE = 512;
const int NUM_INSTANCE = 1024;
const int NUM_OPERATION = 262144;
const int MOD = 10000;
const int MAX_SIZE = 31;

__device__ int gpuIndex(int size, int *keys, int key){
    return (int)(device_find(keys, size, key) - keys);
}

__global__ void procKernel(int n_ins, int *sizes, int *keys, float *values, 
int n_ops, char *ops, int *input_keys, float *input_values, float *ans){
    int b_id = blockIdx.x, b_sz = blockDim.x, t_id = threadIdx.x;
    int global_idx = b_id*b_sz + t_id;
    int sz = sizes[global_idx];
    int start = global_idx*MAX_SIZE;
    for(int i=global_idx*n_ops; i<(global_idx+1)*n_ops; ++i){
        int key = input_keys[i];
        char op = ops[i];
        float value = input_values[i];
        int idx = gpuIndex(sz, keys + start, key);
        if(op == 'g'){
            ans[i] = (idx == sz? 0.:values[start+idx]);
        }
        else if(op == 'r'){
            sz -= (idx != sz);
            keys[start+idx] = keys[start+sz];
            values[start+idx] = values[start+sz];
        }
        else{
            keys[start + idx] = key;
            keys[start + idx] = value;
            sz += (idx==sz && sz < MAX_SIZE);
        }
    }
    sizes[global_idx] = sz;
}

class GPUMapTest{
    int N_ins, N_ops;
    def_dvec(int) dkeys, dinkeys, dsizes;
    def_dvec(float) dvalues, dinvalues;
    def_dvec(char) dops;
public:
    GPUMapTest(int num_ins): N_ins(num_ins){
        dkeys.resize(num_ins * MAX_SIZE);
        dvalues.resize(num_ins * MAX_SIZE);
        dsizes.assign(num_ins, 0);
    }
    void loadOps(const vector<char> &ops, const vector<int> &inkeys,const vector<float> &invals, int n_ops){
        N_ops = n_ops;
        assert((int)ops.size() == N_ops * N_ins);
        dinkeys.resize(N_ops * N_ins);
        dinvalues.resize(N_ops * N_ins);
        dops.resize(N_ops * N_ins);
        gpu_copy(ops, dops);
        gpu_copy(inkeys, dinkeys);
        gpu_copy(invals, dinvalues);
    }
    
    void procOps(vector<float> &ans){
        ans.resize(N_ins * N_ops);
        def_dvec(float) dans(N_ins * N_ops);
        int n_block = (N_ins+BLOCK_SIZE-1)/BLOCK_SIZE;
        procKernel<<<n_block, BLOCK_SIZE>>>(N_ins, to_ptr(dsizes), to_ptr(dkeys), to_ptr(dvalues), N_ops, 
        to_ptr(dops), to_ptr(dinkeys), to_ptr(dinvalues), to_ptr(dans));
        gpu_copy(dans, ans);
        return ;
    }
};


int main(int argc, char *argv[]){
    srand(0);
    int num_ins = NUM_INSTANCE, num_ops = NUM_OPERATION;
    if(argc > 1) num_ins = stoi(argv[1]);
    if(argc > 2) num_ops = stoi(argv[2]);
    /* using cudaEvent to evaluate time */
    cudaEvent_t start, stop;
    float cuda_time;
    cudaEventCreate(&start);   // creating the event 1
    cudaEventCreate(&stop);    // creating the event 2
    
    /* Generating data*/
    cudaEventRecord(start, 0);
    string ref;
    ref += string(1500, 'g') + string(300, 'i') + string(200, 'r');
    vector<char> ops(num_ins * num_ops);
    vector<int> input_keys(num_ins * num_ops);
    vector<float> input_values(num_ins * num_ops);
    generate(input_keys.begin(), input_keys.end(), [](){return rand()%MOD;});
    generate(input_values.begin(), input_values.end(), [](){return float(rand())/RAND_MAX;});
    generate(ops.begin(), ops.end(), [&ref](){return ref[rand()%(int)ref.size()];});
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for generating random data is: "<<cuda_time/1000<<"s"<<endl;
    
    cudaEventRecord(start, 0);
    GPUMapTest gpu_test(num_ins);
    gpu_test.loadOps(ops, input_keys, input_values, num_ops);
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for preparing maps is: "<<cuda_time/1000<<"s"<<endl;
    
    cudaEventRecord(start, 0);
    vector<float> ans;
    gpu_test.procOps(ans);
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for processing operations is: "<<cuda_time/1000<<"s"<<endl;
    
    cout<<"Showing GPU code answers:"<<endl;
    for(int i=0;i<num_ins*num_ops ; i+=num_ins*num_ops/25 + 1) cout<<ans[i]<<' ';
    cout << endl;
    cout<<"DONE!"<<endl;
    
    return 0;
}