#ifndef gdeque_h
#define gdeque_h
#include "glist.h"
#include <cassert>

namespace gpu_stl{
    /*
    Here I implement a gpu version of stl deque using linked list (pointer version),
    which contains member functions: (do not have at() and max_size())
        empty();
        size();
        front();
        back();
        push_front(g);
        push_back(g);
        pop_front();
        pop_back();
    */
    template<typename T>
    struct deque{
        gpu_stl::list<T> container;
    public:
        __device__ deque(){}
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
        __device__ void push_front(const T&val){
            container.push_front(val);
        }
        __device__ void push_back(const T&val){
            container.push_back(val);
        }
        __device__ void pop_front(){
            container.pop_front();
        }
        __device__ void pop_back(){
            container.pop_back();
        }
    };
}

#endif