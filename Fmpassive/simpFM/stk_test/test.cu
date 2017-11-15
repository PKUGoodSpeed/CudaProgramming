#include <iostream>
#include "../include/gpu_stack.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;

const int STACK_SIZE = 100;

__global__ void test(float *output){
    gpu_stack<float, STACK_SIZE> stk;
    for(int i=1;i<=STACK_SIZE;++i){
        stk.push(1.5*i);
    }
    int idx = 0, k = 0;
    while(!stk.empty()){
        stk.pop_k(k);
        if(stk.empty()) return;
        output[idx] = stk.top();
        idx += 1;
        output[idx] = (float)stk.size();
        k += 1;
        idx += 1;
    }
}

int main(){
    def_dvec(float) dev_out(40, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}