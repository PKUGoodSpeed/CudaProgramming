#include <iostream>
#include "../include/gpu_list.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;


__global__ void test(float *output){
    gpu_list<float> list;
    for(int i=0;i<20;++i) list.push_front(i*1.7);
    for(int i=20;i<40;++i) list.push_back(i*1.7);
    for(auto p=list.begin(); p!=list.end(); ++p){
        list.insert(p, 10086.);
    }
    list.insert(list.begin(), 111);
    //list.back() = 100.;
    //list.front() = 200.;
    int idx = 0;
    for(auto p=list.begin(); p!=list.end(); ++p){
        output[idx++] = (*p);
    }
    output[idx++] = list.front();
    output[idx++] = *list.begin();
    output[idx++] = (float)list.size();
}

int main(){
    def_dvec(float) dev_out(83, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}