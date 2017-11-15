#ifndef gpu_map_h
#define gpu_map_h
#include <cassert>
#include "gpu_stack.h"

template<typename KEY_T, typename VAL_T>
struct gpu_pair{
    KEY_T first;
    VAL_T second;
public:
    __device__ gpu_pair(const KEY_T &key = 0, const VAL_T &val = 0){
        this->first = key;
        this->second = val;
    }
};

template<typename KEY_T, typename VAL_T, const int MAX_MAP_SIZE>
struct gpu_map{
    gpu_stack<int, MAX_MAP_SIZE> index_pool;
    gpu_pair<KEY_T, VAL_T> container[MAX_MAP_SIZE];
    bool exist[MAX_MAP_SIZE];
public:
    /* Initialization */
    __device__ gpu_map(){
        memset(exist, false, sizeof(exist));
        for(int i=0;i<MAX_MAP_SIZE;++i) index_pool.push(i);
    }
    
    /* empty */
    __device__ bool empty(){
        return index_pool.size() == MAX_MAP_SIZE;
    }
    
    /* full */
    __device__ bool full(){
        return index_pool.empty();
    }
    
    /* size */
    __device__ int size(){
        return MAX_MAP_SIZE - index_pool.size();
    }
    
    /* find */
    __device__ int find(const KEY_T &key){
        for(int i=0;i<MAX_MAP_SIZE;++i) if(exist[i] && container[i].first == key) return i;
        return -1;
    }
    
    /* find value */
    __device__ int rfind(const VAL_T &val){
        for(int i=0;i<MAX_MAP_SIZE;++i) if(exist[i] && container[i].second == val) return i;
        return -1;
    }
    
    /* given key return value */
    __device__ VAL_T get(const KEY_T &key){
        int find_rst = this->find(key);
        if(find_rst == -1) return -1; /* invalid return */
        return container[find_rst].second;
    }
    
    /* given value return the first key */
    __device__ KEY_T rget(const VAL_T &val){
        int find_rst = this->rfind(val);
        if(find_rst == -1) return -1; /* invalid return */
        return container[find_rst].first;
    }
    
    /* set or insert key, value pair */
    __device__ void set(const KEY_T &key, const VAL_T &val){
        int find_rst = this->find(key);
        if(find_rst != -1){
            container[find_rst].second = val;
            return;
        }
        assert(!this->full());
        int idx = index_pool.top();
        index_pool.pop();
        container[idx].first = key;
        container[idx].second = val;
        exist[idx] = true;
        return;
    }
    
    /* erase a key */
    __device__ void erase(const KEY_T& key){
        int find_rst = this->find(key);
        if(find_rst == -1) return;
        assert(!this->empty());
        index_pool.push(find_rst);
        exist[find_rst] = false;
        return ;
    }
};

#endif