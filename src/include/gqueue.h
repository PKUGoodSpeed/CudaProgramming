#ifndef gqueue_h
#define gqueue_h
#include "glist.h"
#include <cassert>

namespace gpu_stl{
    /*
    Here I implement a gpu version of stl queue using linked list (pointer version),
    which contains member functions:
        empty();
        size();
        front();
        back();
        push(g);
        pop();
    */
    template<typename T>
    struct queue{
        gpu_stl::list<T> container;
    public:
        __device__ queue(){}
        __device__ bool empty(){
            return container.empty();
        }
        __device__ int size(){
            return container.size();
        }
        __device__ T front(){
            assert(!container.empty());
            return container.front();
        }
        __device__ T back(){
            assert(!container.empty());
            return container.back();
        }
        __device__ void push(const T&val){
            container.push_back(val);
        }
        __device__ void pop(){
            container.pop_front();
        }
    };
}

#endif