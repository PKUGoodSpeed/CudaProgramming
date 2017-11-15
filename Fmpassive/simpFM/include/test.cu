#include <cassert>
#include "gpu_integer_queue.cu"
#include "gpu_integer_stack.cu"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])

#define SET_SIZE 40
#define SET_ARRAY_SIZE 41
using namespace std;

struct gpu_integer_set{
    def_gpu_stack(idxs);
    int gset[SET_SIZE];
public:
    __device__ gpu_integer_set(){
        gpu_stack_init(this->idxs);
    }
    __device__ void push(int x){
        gpu_stack_push(this->idxs, x);
    }
    __device__ int top(){
        return gpu_stack_top(this->idxs);
    }
    __device__ void pop(){
        gpu_stack_pop(this->idxs);
    }
    __device__ bool empty(){
        return gpu_stack_empty(this->idxs);
    }
};

__global__ void test(int *output){
    gpu_integer_set S;
    for(int i=1;i<=STACK_SIZE;++i) S.push(i);
    int idx = 0;
    while(!S.empty()){
        output[idx] = S.top();
        S.pop();
        idx += 1;
    }
}

int main(){
    def_dvec(int) dev_out(40, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}
