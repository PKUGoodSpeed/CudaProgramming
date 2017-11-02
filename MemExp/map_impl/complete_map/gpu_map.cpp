/*
 We consider 1024 map, which are initially constructed with (at most)size = 4096 key, value pairs, for which keys are integers and values are float point numbers.
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
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int BLOCK_SIZE = 1;

const int NUM_INSTANCE = 1024;
const int NUM_OPERATION = 262144;
const int MAX_MAP_SIZE = 4096;
const int MOD = 10000;

__device__ void cudaInsert(int *size, int *keys, float *values, int key, float value){
    if((*size) == MAX_MAP_SIZE) return ;
    int sz = *size;
    int idx = int(thrust::find(thrust::device, keys, keys+sz, key) - keys);
    if(idx < sz) return ;
    keys[sz] = key;
    values[sz] = value;
    (*size) += 1;
}

__device__ void cudaRemove(int *size, int *keys, float *values, int key){
    int sz = *size;
    int idx = int(thrust::find(thrust::device, keys, keys+sz, key) - keys);
    if(idx == sz) return;
    keys[idx] = keys[sz - 1];
    values[idx] = values[sz - 1];
    (*size) -= 1;
}

__device__ void cudaModify(int size, int *keys, float *values, int key, float value){
    int idx = int(thrust::find(thrust::device, keys, keys+size, key) - keys);
    if(idx < size){
        values[idx] = value;
        keys[idx] = key;
    }
}

__device__ bool cudaSearch(int size, int *keys, int key){
    int idx = int(thrust::find(thrust::device, keys, keys+size, key) - keys);
    return idx<size;
}

__device__ float cudaGetValue(int size, int *keys, float *values, int key){
    int idx = int(thrust::find(thrust::device, keys, keys+size, key) - keys);
    if(idx < size) return values[idx];
    return 0.;
}

__device__ int cudaGetSize(int size){
    return size;
}

__device__ bool cudaIsEmpty(int size){
    return !size;
}

__device__ bool cudaIsFull(int size){
    return size == MAX_MAP_SIZE;
}

/* 
here we assume that the operation is only 'g', which is getting values.
Currently we are only testing the case that do not need synchronization
*/
__global__ void cudaProcKernel(int n_ins, int *sizes, int *keys, float *values, 
int n_ops, char *ops, int *input_keys, float *input_values, float *ans){
    int b_idx = blockIdx.x;
    int local_keys[MAX_MAP_SIZE];
    float local_values[MAX_MAP_SIZE];
    int map_start = MAX_MAP_SIZE * b_idx, ops_start = n_ops * b_idx, size = sizes[b_idx];;
    for(int i=0;i<size;++i){
        local_keys[i] = keys[map_start + i];
        local_values[i] = values[map_start + i];
    }
    for(int i=0;i<n_ops;++i){
        int j = ops_start + i;
        char c = ops[j];
        int key = input_keys[j];
        float value = input_values[j];
        if(c == 'i') cudaInsert(&size, local_keys, local_values, key, value);
        else if(c == 'r') cudaRemove(&size, local_keys, local_values, key);
        else if(c == 'm') cudaModify(size, local_keys, local_values, key, value);
        else if(c == 's') ans[j] = (float)cudaSearch(size, local_keys, key);
        else if(c == 'g') ans[j] = cudaGetValue(size, local_keys, local_values, key);
        else if(c == 'z') ans[j] = (float)size;
        else if(c == 'e') ans[j] = (float)(!size);
        else if(c == 'f') ans[j] = (float)(size == MAX_MAP_SIZE);
        else ans[j] = 0.;
    }
    for(int i=0;i<size;++i){
        keys[map_start + i] = local_keys[i];
        values[map_start + i] = local_values[i];
    }
    sizes[b_idx] = size;
    return ;
}

// CPU version
class GPUMapTest{
    int N_ins, N_ops;
    def_dvec(int) dkeys, dinkeys, dsizes;
    def_dvec(float) dvalues, dinvalues;
    def_dvec(char) dops;
public:
    GPUMapTest(int num_ins): N_ins(num_ins){
        dkeys.resize(num_ins * MAX_MAP_SIZE);
        dvalues.resize(num_ins * MAX_MAP_SIZE);
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
        cudaProcKernel<<<N_ins, BLOCK_SIZE>>>(N_ins, to_ptr(dsizes), to_ptr(dkeys), to_ptr(dvalues), N_ops, 
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
    ref += string(500, 'g') + string(500, 's') + string(200, 'i') + string(200, 'm') + "e" + "zz" + "f" + string(100, 'r');
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

