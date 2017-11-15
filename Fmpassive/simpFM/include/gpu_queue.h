#ifndef gpu_queue_h
#define gpu_queue_h
#include <cassert>

template<typename T, const int MAX_QUEUE_SIZE = 50>
struct gpu_queue{
    int i_start, i_end;
    T container[MAX_QUEUE_SIZE];
public:
    /* Initialize the queue */
    __device__ gpu_queue() {
        i_start = i_end = 0;
    }
    
    /* empty */
    __device__ bool empty(){
        return i_start >= i_end;
    }
    
    /* size */
    __device__ int size(){
        return i_end - i_start;
    }
    
    /* get front */
    __device__ T front(){
        assert(!this->empty());
        return container[i_start%MAX_QUEUE_SIZE];
    }
    
    /* get back */
    __device__ T back(){
        assert(!this->empty());
        return container[(i_end-1)%MAX_QUEUE_SIZE];
    }
    
    /* push */
    __device__ void push(const T &val){
        assert(this->size() < MAX_QUEUE_SIZE);
        container[i_end%MAX_QUEUE_SIZE] = val;
        ++i_end;
    }
    
    /* pop */
    __device__ void pop(){
        assert(!this->empty());
        ++i_start;
    }
    
    /* pop k elements */
    __device__ void pop_k(const int &k){
        i_start += min(k, this->size());
    }
};

#endif