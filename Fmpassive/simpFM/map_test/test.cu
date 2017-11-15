#include <iostream>
#include "../include/gpu_map.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;

const int MAP_SIZE = 100;

__global__ void test(float *output){
    gpu_map<int, float, MAP_SIZE> map;
    for(int i=0;i<MAP_SIZE;++i){
        map.set(2*i, 1.7*i);
    }
    int idx = 0;
    for(int i=0;i<100;++i){
        output[idx++] = map.get(i);
        output[idx++] = (int)map.rget(1.7*i);
        map.erase(i);
        if(i%2) map.set(i, i*1.7);
    }
    for(int i=0;i<100;++i) {
        output[idx++] = map.get(i);
        output[idx++] = (int)map.rget(1.7*i);
    }
}

int main(){
    def_dvec(float) dev_out(400, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}