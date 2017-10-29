/*
 We consider 1024 map, which are initially constructed with (at most)size = 4096 key, value pairs, for which keys are integers and values are float point numbers.
 Then each of them processes 262,144 operations, which only include 'g': search/getting values (If did not find, return -1.), 'e': check empty, 'z': check size.
 */
#include <bits/stdc++.h>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int BLOCK_SIZE = 1024;

const int NUM_INSTANCE = 1024;
const int NUM_OPERATION = 262144;
const int MAX_MAP_SIZE = 4096;
const int MOD = 10000;

__device__ float cuMapGetValue(int size, int *keys, float *values, int key){
    for(int i=0;i<size;++i) if(keys[i] == key) return values[i];
    return -1.;
}

/* 
here we assume that the operation is only 'g', which is getting values.
Currently we are only testing the case that do not need synchronization
*/
__global__ void cudaProcKernel(int n_ins, int *sizes, int *keys, float *values, int n_ops, int *input, float *ans){
    int b_sz = blockDim.x, b_idx = blockIdx.x, t_idx = threadIdx.x;
    __shared__ int local_keys[MAX_MAP_SIZE];
    __shared__ float local_values[MAX_MAP_SIZE];
    int map_size = sizes[b_idx], map_start = MAX_MAP_SIZE * b_idx, ops_start = n_ops * b_idx;
    assert(map_size <= MAX_MAP_SIZE);
    for(int i=t_idx ; i<map_size ; i+=b_sz){
        local_keys[i] = keys[map_start + i];
        local_values[i] = values[map_start + i];
    }
    __syncthreads();
    for(int i=t_idx;i<n_ops;i+=b_sz){
        ans[ops_start + i] = cuMapGetValue(map_size, local_keys, local_values, input[ops_start + i]);
    }
    return ;
}

// CPU version
class GPUMapTest{
    int N_ins, N_ops;
    def_dvec(int) dkeys, dinput, dsizes;
    def_dvec(float) dvalues;
public:
    GPUMapTest(const vector<vector<int>> &keys, const vector<vector<float>> &values): N_ins((int)keys.size()){
        assert((int)values.size() == N_ins);
        dkeys.resize(N_ins * MAX_MAP_SIZE);
        dvalues.resize(N_ins * MAX_MAP_SIZE);
        dsizes.resize(N_ins);
        for(int i=0;i<N_ins;++i){
            gpu_copy_to(keys[i], dkeys, i*MAX_MAP_SIZE);
            gpu_copy_to(values[i], dvalues, i*MAX_MAP_SIZE);
            dsizes[i] = (int)keys[i].size();
        }
    }
    void loadOps(const vector<int> &input, int n_ops){
        N_ops = n_ops;
        assert((int)input.size() == N_ops * N_ins);
        dinput.resize(N_ops * N_ins);
        gpu_copy(input, dinput);
    }
    
    void procOps(vector<float> &ans){
        ans.resize(N_ins * N_ops);
        def_dvec(float) dans(N_ins * N_ops);
        cudaProcKernel<<<N_ins, BLOCK_SIZE>>>(N_ins, to_ptr(dsizes), to_ptr(dkeys), to_ptr(dvalues), N_ops, to_ptr(dinput), to_ptr(dans));
        gpu_copy(dans, ans);
        return ;
    }
};

int main(int argc, char *argv[]){
    int size = MAX_MAP_SIZE, num_ins = NUM_INSTANCE, num_ops = NUM_OPERATION;
    if(argc > 1) num_ins = stoi(argv[1]);
    if(argc > 2) num_ops = stoi(argv[2]);
    /* using cudaEvent to evaluate time */
    cudaEvent_t start, stop;
    float cuda_time;
    cudaEventCreate(&start);   // creating the event 1
    cudaEventCreate(&stop);    // creating the event 2
    
    /* Generating data*/
    cudaEventRecord(start, 0);
    vector<vector<int>> keys(num_ins);
    vector<vector<float>> values(num_ins);
    vector<int> input(num_ins * num_ops);
    for(int i=0;i<num_ins;++i){
        unordered_set<int> exist;
        for(int j=0;j<size;++j){
            int tmp_key = rand()%MOD;
            if(!exist.count(tmp_key)){
                keys[i].push_back(tmp_key);
                values[i].push_back(float(rand())/RAND_MAX);
                exist.insert(tmp_key);
            }
        }
    }
    generate(input.begin(), input.end(), [](){return rand()%MOD;});
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for generating random data is: "<<cuda_time/1000<<"s"<<endl;
    
    cudaEventRecord(start, 0);
    GPUMapTest gpu_test(keys, values);
    gpu_test.loadOps(input, num_ops);
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

