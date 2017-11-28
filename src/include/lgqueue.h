#ifndef lgqueue_h
#define lgqueue_h
#include <cassert>

namespace gpu_linearized_stl{
    /*
    Here we use array to implement the queue which can be used on GPU.
    This stack contains the following member functions:
        empty();
        size();
        max_size();
        front();
        back();
        push(g);
        pop();
        pop_k();
    */
    template<typename T, const int MAX_SIZE=128>
    struct queue{
        T container[MAX_SIZE];
        int start, end;
    public:
        __device__ queue(): start(0), end(0) {}
        __device__ bool empty(){
            return start == end;
        }
        __device__ int size(){
            return end - start;
        }
        __device__ int max_size(){
            return MAX_SIZE;
        }
        __device__ T front(){
            assert(end > start);
            return container[start%MAX_SIZE];
        }
        __device__ T back(){
            assert(end > start);
            return container[(end-1)%MAX_SIZE];
        }
        __device__ void push(const T&val){
            assert(this->size() < MAX_SIZE);
            container[end%MAX_SIZE] = val;
            ++end;
        }
        __device__ void pop(){
            assert(end > start);
            ++start;
        }
        __device__ void pop_k(const int&k){
            start = min(end, start+k);
        }
        __device__ void clear(){
            start = end = 0;
        }
    };
}

#endif