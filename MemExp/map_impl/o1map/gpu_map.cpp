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
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define gpu_seq(x) thrust::sequence((x).begin(), (x).end())
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int BLOCK_SIZE = 1024;
const int NUM_INSTANCE = 1024;
const int MAX_MAP_SIZE = 1024;
const int NUM_OPERATION = 262144;
const int MOD = 10000;

__global__ void procKernel(int n_ins, int *sizes, int *keys, float *values, 
int n_ops, char *ops, int *input_keys, float *input_values, float *ans){
    int b_id = blockIdx.x, b_sz = blockDim.x, t_id = threadIdx.x;
    int local_key;
    float local_val;
    __shared__ int int_info[3];
    /*
    0: size
    1: index
    2: key
    */
    __shared__ float float_info[1]; /* value*/
    __shared__ char char_info[1] /*op type*/
    __shared__ bool bool_info[1] /* whether updated*/
    if(!t_id) int_info[0] = sizes[b_id];
    __syncthreads();
    if(t_id < int_info[0]){
        local_key = keys[t_id];
        local_val = values[t_id];
    }
    __syncthreads();
    for(int q=0; q<n_ops; ++q){
        if(!t_id){
            int_info[2] = input_keys[b_id* n_ops + q];
            float_info[0]= input_values[b_id* n_ops + q];
            char_info[0] = ops[b_id* n_ops + q];
            bool_info[0] = false;
        }
        __syncthreads();
        if(char_info[0] == 'i'){
            if(local_key == int_info[2]){
        }
    }
}


int main(){
    
    return 0;
}