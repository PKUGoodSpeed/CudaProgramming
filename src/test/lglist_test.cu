#include <iostream>
#include "../include/lglist.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;


__global__ void test(float *output){
    gpu_linearized_stl::list<float,100> list;
    int idx = 0;
    output[idx++] = list.max_size();
    output[idx++] = list.full();
    output[idx++] = (float)list.empty();
    output[idx++] = (float)list.size();
    for(int i=0;i<10;++i){
        list.push_back(i*1.7);
        output[idx++] = (float)list.empty();
        output[idx++] = (float)list.size();
    }
    for(int i=0;i<6;++i) {
        int a = list.end();
        auto p = list.insert(list.decrement(a), 55);
        list.decrement(p);
        list.decrement(p);
        list.insert(p, 77);
    }
    for(auto p=list.begin(); p!=list.end();list.increment(p)) output[idx++] = list.at(p);
    output[idx++] = list.front();
    output[idx++] = list.back();
    int p;
    while((p=list.find(77))!=list.end()){
        list.erase(p);
    }
    output[idx++] = 10086;
    for(auto p=list.begin(); p!=list.end();list.increment(p)) output[idx++] = list.at(p);
    list.reverse();
    output[idx++] = 10086;
    for(auto p=list.begin(); p!=list.end();list.increment(p)) output[idx++] = list.at(p);
    list.pop_front();
    list.pop_back();
    list.pop_back();
    output[idx++] = 10086;
    for(auto p=list.begin(); p!=list.end();list.increment(p)) output[idx++] = list.at(p);
}

int main(){
    def_dvec(float) dev_out(150, 0);
    test<<<1, 1>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}
