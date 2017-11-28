#ifndef gpu_set_h
#define gpu_set_h
#include <cassert>
#include "gpu_stack.h"

template<typename T, const int MAX_SET_SIZE = 50>
struct gpu_set{
    gpu_stack<int, MAX_SET_SIZE> index_pool;
    T container[MAX_SET_SIZE];
    bool exist[MAX_SET_SIZE];
public:
    /* Initialization */
    __device__ gpu_set(){
        memset(exist, false, sizeof(exist));
        for(int i=0;i<MAX_SET_SIZE;++i) index_pool.push(i);
    }
    
    /* empty */
    __device__ bool empty(){
        return index_pool.size() == MAX_SET_SIZE;
    }
    
    /* full */
    __device__ bool full(){
        return index_pool.empty();
    }
    
    /* size */
    __device__ int size(){
        return MAX_SET_SIZE - index_pool.size();
    }
    
    /* find */
    __device__ int find(const T &target){
        for(int i=0;i<MAX_SET_SIZE;++i) if(exist[i] && container[i] == target) return i;
        return -1;
    }
    
    /* insert */
    __device__ void insert(const T &target){
        int find_rst = this->find(target);
        if(find_rst != -1) return;
        assert(!this->full());
        int idx = index_pool.top();
        index_pool.pop();
        container[idx] = target;
        exist[idx] = true;
    }
    
    /* erase */
    __device__ void erase(const T &target){
        int find_rst = this->find(target);
        if(find_rst == -1) return;
        assert(!this->empty());
        index_pool.push(find_rst);
        exist[find_rst] = false;
    }
};

#endif