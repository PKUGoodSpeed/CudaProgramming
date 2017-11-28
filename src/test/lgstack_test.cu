#include <iostream>
#include "../include/lgstack.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;


__global__ void test(float *output){
    gpu_linearized_stl::stack<float, 40> stk;
    int idx = 0;
    output[idx++] = stk.max_size();
    output[idx++] = stk.empty();
    output[idx++] = stk.size();
    output[idx++] = 10086;
    for(int i=1;i<=20;++i){
        stk.push(i*1.7);
        output[idx++] = stk.empty();
        output[idx++] = stk.size();
    }
    output[idx++] = 10086;
    while(!stk.empty()){
        output[idx++] = stk.empty();
        output[idx++] = stk.size();
        output[idx++] = stk.top();
        stk.pop();
    }
}

int main(){
    def_dvec(float) dev_out(120, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}
