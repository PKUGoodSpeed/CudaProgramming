#ifndef gpu_stack_h
#define gpu_stack_h
#include <cassert>

template<typename T, const int MAX_STACK_SIZE = 50>
struct gpu_stack{
    int stk_size;
    T container[MAX_STACK_SIZE];
public:
    /* Initialize the stack */
    __device__ gpu_stack(){
        stk_size = 0;
    }
    
    /* empty */
    __device__ bool empty(){
        return !stk_size;
    }
    
    /* size */
    __device__ int size(){
        return stk_size;
    }
    
    /* get top */
    __device__ T top(){
        assert(!this->empty());
        return container[stk_size-1];
    }
    
    /* push */
    __device__ void push(const T &val){
        assert(stk_size < MAX_STACK_SIZE);
        container[stk_size] = val;
        ++stk_size;
    }
    
    /* pop */
    __device__ void pop(){
        assert(stk_size);
        --stk_size;
    }
    
    /* pop k elements */
    __device__ void pop_k(const int &k){
        stk_size -= min(k, stk_size);
    }
};

#endif