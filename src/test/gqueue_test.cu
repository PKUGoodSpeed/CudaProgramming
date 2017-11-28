#include <iostream>
#include "../include/gqueue.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;


__global__ void test(float *output){
    gpu_stl::queue<float> que;
    int idx = 0;
    output[idx++] = que.empty();
    output[idx++] = que.size();
    output[idx++] = 10086;
    for(int i=1;i<=15;++i){
        que.push(i*1.7);
        output[idx++] = que.empty();
        output[idx++] = que.size();
    }
    output[idx++] = 10086;
    while(!que.empty()){
        output[idx++] = que.empty();
        output[idx++] = que.size();
        output[idx++] = que.front();
        output[idx++] = que.back();
        que.pop();
    }
}

int main(){
    def_dvec(float) dev_out(120, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}
