#include <iostream>
#include "../include/gpu_integer_queue.cu"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;

__global__ void test(int *output){
    def_gpu_queue(que);
    gpu_queue_init(que);
    for(int i=0;i<QUEUE_SIZE;++i){
        gpu_queue_push(que, i);
    }
    int idx = 0, k = 0;
    while(!gpu_queue_empty(que)){
        gpu_queue_pop_k(que, min(k, gpu_queue_size(que)));
        output[idx] = gpu_queue_front(que);
        k += 1;
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