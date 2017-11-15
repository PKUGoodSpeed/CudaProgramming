#include <cassert>

#define STACK_SIZE 40
#define STACK_ARRAY_SIZE 41
#define def_gpu_stack(x) int (x)[STACK_ARRAY_SIZE]

using namespace std;

/* Initialize the stack */
__device__ void gpu_stack_init(int *stk){
    stk[STACK_SIZE] = 0;
}

/* Check whether a stack is empty or not */
__device__ bool gpu_stack_empty(int *stk){
    return stk[STACK_SIZE] == 0;
}

/* Get the size of the stack */
__device__ int gpu_stack_size(int *stk){
    return stk[STACK_SIZE];
}

/* Get the top of the stack */
__device__ int gpu_stack_top(int *stk){
    assert(!gpu_stack_empty(stk));
    return stk[stk[STACK_SIZE] - 1];
}

/* Push a variable from the top of the stack */
__device__ void gpu_stack_push(int *stk, int val){
    assert(gpu_stack_size(stk) < STACK_SIZE);
    stk[stk[STACK_SIZE]] = val;
    stk[STACK_SIZE] += 1;
}

/* Pop a value from the top of the stack */
__device__ void gpu_stack_pop(int *stk){
    assert(!gpu_stack_empty(stk));
    stk[STACK_SIZE] -= 1;
}

/* Pop k variables from the top of the stack */
__device__ void gpu_stack_pop_k(int *stk, int k){
    assert(k <= gpu_stack_size(stk));
    stk[STACK_SIZE] -= k;
}