#ifndef gstack_h
#define gstack_h
#include "glist.h"
#include <cassert>

namespace gpu_stl{
    /*
    Here I implement a gpu version of stl stack using linked list (pointer version),
    which contains member functions:
        empty();
        size();
        top();
        push(g);
        pop();
    */
    template<typename T>
    struct stack{
        gpu_stl::list<T> container;
    public:
        __device__ stack(){}
        __device__ bool empty(){
            return container.empty();
        }
        __device__ int size(){
            return container.size();
        }
        __device__ T top(){
            assert(!container.empty());
            return container.back();
        }
        __device__ void push(const T&val){
            container.push_back(val);
        }
        __device__ void pop(){
            container.pop_back();
        }
    };
}

#endif