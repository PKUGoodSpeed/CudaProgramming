#include <iostream>
#include "../include/lglist.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
using namespace std;

const int MAX_LENGTH = 170;

__device__ void printList(gpu_linearized_stl::list<float, MAX_LENGTH> &l){
    if(l.empty()){
        printf("\n===== Empty list =====\n\n");
        return;
    }
    printf("\n===== Showing the list =====\n");
    printf("list size = %d \n", l.size());
    for(auto i=l.begin(); i!=l.end(); l.increment(i)){
        printf("%.2f ", l.at(i));
    }
    printf("\n\n");
}


__global__ void test(float *output){
    gpu_linearized_stl::list<float, MAX_LENGTH> list;
    printList(list);
    //
    for(int i=0;i<5;++i) list.push_back(i*1.7);
    printf("for(int i=0;i<5;++i) list.push_back(i*1.7);");
    printList(list);
    //
    for(int i=6;i<10;++i) list.push_front(i*2.5);
    printf("for(int i=6;i<11;++i) list.push_front(i*2.5);");
    printList(list);
    //
    for(int i=0;i<2;++i) list.pop_back();
    printf("for(int i=0;i<2;++i) list.pop_back();");
    printList(list);
    
    //
    for(int i=0;i<2;++i) list.pop_front();
    printf("for(int i=0;i<2;++i) list.pop_front();");
    printList(list);
    //
    printf("list.at(list.begin())\n");
    printf("%.2f\n\n", list.at(list.begin()));
    //
    printf("list.front()\n");
    printf("%.2f\n\n", list.front());
    //
    auto idx = list.end();
    printf("auto idx = list.end(); list.at(list.decrement(idx))\n");
    printf("%.2f\n\n", list.at(list.decrement(idx)));
    //
    printf("list.back()\n");
    printf("%.2f\n\n", list.back());
    //
    list.reverse();
    printf("list.reverse();");
    printList(list);
    //
    list.insert(list.begin(), 3.14);
    printf("list.insert(list.begin(), 3.14);");
    printList(list);
    //
    list.insert(list.end(), 3.14);
    printf("list.insert(list.end(), 3.14);");
    printList(list);
    // 
    list.erase(list.begin());
    printf("list.erase(list.begin());");
    printList(list);
    //
    printf("list.at(list.begin())\n");
    printf("%.2f\n\n", list.at(list.begin()));
    //
    list.erase(list.end());
    printf("list.erase(list.end());");
    printList(list);
    //
    printf("list.back()\n");
    printf("%.2f\n\n", list.back());
    //
    for(auto p=list.begin(); p!=list.end(); list.increment(p)) list.insert(p, 22.22);
    printf("for(auto p=list.begin(); p!=list.end(); list.increment(p)) list.insert(p, 22.22);");
    printList(list);
    //
    int p;
    while((p=list.find(22.22))!=list.end()) list.erase(p);
    printf("while((auto p=list.find(22.22))!=list.end()) list.erase(p);");
    printList(list);
    
    gpu_linearized_stl::list<float, MAX_LENGTH> list2;
    list2.push_back(list.at(list.begin()));
    list.at(list.begin()) = 10086;
    printf("Reference checking");
    printList(list);
    printList(list2);
}

int main(){
    def_dvec(float) dev_out(1, 0);
    test<<<1, 2>>>(to_ptr(dev_out));
    return 0;
}