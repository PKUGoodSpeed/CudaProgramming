#include <iostream>
#include "../include/gpu_list.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;


__global__ void test(float *output){
    gpu_list<float> list;
    for(int i=0;i<80;++i) list.push_back(float(i));
    int idx = 0;
    list.reverse();
    list.sort();
    for(auto p=list.begin(); p!=list.end(); ++p) output[idx++] = *p;
    output[idx++] = list.front();
    output[idx++] = *list.begin();
    output[idx++] = (float)list.size();
}

int main(){
    def_dvec(float) dev_out(100, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}
