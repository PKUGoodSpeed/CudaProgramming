#include <iostream>
#include "../include/glist.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;


__global__ void test(float *output){
    gpu_stl::list<float> list;
    int idx = 0;
    output[idx++] = (float)list.empty();
    output[idx++] = (float)list.size();
    for(int i=0;i<10;++i){
        list.push_back(i*1.7);
        output[idx++] = (float)list.empty();
        output[idx++] = (float)list.size();
    }
    for(int i=0;i<6;++i) {
        auto p = list.insert(--list.end(), 55);
        ++p;
        ++p;
        list.insert(p, 77);
    }
    for(auto p=list.begin(); p!=list.end();++p) output[idx++] = *p;
    output[idx++] = list.front();
    output[idx++] = *list.begin();
    *list.begin() = 3.1415926;
    output[idx++] = list.front();
    output[idx++] = *list.begin();
    output[idx++] = 10086;
    output[idx++] = list.back();
    output[idx++] = *(--list.end());
    *(--list.end()) = 3.124235;
    output[idx++] = list.back();
    output[idx++] = *(--list.end());
    gpu_stl::list<float>::iterator p;
    while((p=list.find(77))!=list.end()){
        list.erase(p);
    }
    output[idx++] = 10086;
    for(auto p=list.begin(); p!=list.end();++p) output[idx++] = *p;
    list.reverse();
    output[idx++] = 10086;
    for(auto p=list.begin(); p!=list.end();++p) output[idx++] = *p;
    list.pop_front();
    list.pop_back();
    output[idx++] = 10086;
    for(auto p=list.begin(); p!=list.end();++p) output[idx++] = *p;
    gpu_stl::list<float>::iterator p1(--list.end());
    output[idx++] = 10086;
    output[idx++] = *p1;
    list.clear();
    output[idx++] = 10086;
    output[idx++] = list.empty();
    output[idx++] = 10086;
    output[idx++] = list.size();
    output[idx++] = max(123141, 2335436);
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(){
    def_dvec(float) dev_out(120, 0);
    test<<<10, 10>>>(to_ptr(dev_out));
    for(auto k:dev_out) cout<<k<<' ';
    cout<<endl;
    return 0;
}
