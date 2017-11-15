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
};

__global__ void test(int *output){
    gpu_integer_set S;
    for(int i=1;i<=STACK_SIZE;++i) gpu_stack_push(S.idxs, i);
    int idx = 0;
    while(!gpu_stack_empty(S.idxs)){
        output[idx] = gpu_stack_top(S.idxs);
        gpu_stack_pop(S.idxs);
    }
}

int main(){
    def_dvec(int) dev_out(40, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}
