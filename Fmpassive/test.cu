#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

__global__ void cudaTestSum(int *ans){
    int b_sz = blockDim.x, b_id = blockIdx.x, t_id = threadIdx.x;
    __shared__ int sum;
    if(!t_id){
        sum = 0;
    }
    __syncthreads();
    if(t_id < 500){
        for(int i=1;i<=10;++i){
            atomicAdd(&sum, i);
        }   
    }
    __syncthreads();
    if(!t_id){
        ans[b_id] = sum;
    }
}

int main(int argc, char* argv[]){
    int N = 1024, block_size = 1000;
    if(argc > 1) N = stoi(argv[1]);
    if(argc > 2) block_size = stoi(argv[2]);
    def_dvec(int) ans(N, 0);
    cudaTestSum<<<N, block_size>>>(to_ptr(ans));
    for(auto k:ans) cout<<k<<' ';
    return 0;
}