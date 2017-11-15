#include <iostream>
#include <set>
#include "../include/gpu_set.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;

const int SET_SIZE = 100;

__global__ void test(int *output){
    gpu_set<int, SET_SIZE> set;
    for(int i=0;i<SET_SIZE;++i){
        set.insert(2*i);
    }
    int idx = 0;
    for(int i=0;i<100;++i){
        output[idx++] = i;
        output[idx++] = set.find(i);
        set.erase(i);
        if(i%2) set.insert(i);
    }
    for(int i=0;i<100;++i) {
        output[idx++] = i;
        output[idx++] = set.find(i);
    }
}

int main(){
    def_dvec(int) dev_out(400, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}