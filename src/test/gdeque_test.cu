#include <iostream>
#include "../include/gdeque.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;


__global__ void test(float *output){
    gpu_stl::deque<float> deque;
    int idx = 0;
    output[idx++] = deque.empty();
    output[idx++] = deque.size();
    output[idx++] = 10086;
    for(int i=1;i<=15;++i){
        if(i%2) deque.push_front(i*1.7);
        else deque.push_back(i*1.7);
        output[idx++] = deque.empty();
        output[idx++] = deque.size();
    }
    output[idx++] = 10086;
    while(!deque.empty()){
        output[idx++] = deque.empty();
        output[idx++] = deque.size();
        output[idx++] = deque.front();
        output[idx++] = deque.back();
        deque.pop_front();
    }
}

int main(){
    def_dvec(float) dev_out(120, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}
