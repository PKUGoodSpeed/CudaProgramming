#include <cassert>

#define QUEUE_SIZE 40
#define QUEUE_ARRAY_SIZE 42
#define def_gpu_queue(x) int (x)[QUEUE_ARRAY_SIZE]

using namespace std;

/* Initialize the starting and ending position */
__device__ void gpu_queue_init(int *que){
    que[QUEUE_SIZE] = que[QUEUE_SIZE+1] = 0;
}

/* Check whether a que is empty or not */
__device__ bool gpu_queue_empty(int *que){
    return que[QUEUE_SIZE] >= que[QUEUE_SIZE+1];
}

/* Get the front integer of the queue */
__device__ int gpu_queue_front(int *que){
    assert(!gpu_queue_empty(que));
    return que[que[QUEUE_SIZE]%QUEUE_SIZE];
}

/* Get the back integer of the queue */
__device__ int gpu_queue_back(int *que){
    assert(!gpu_queue_empty(que));
    return que[(que[QUEUE_SIZE+1]-1)%QUEUE_SIZE];
}

/* Return the size of the queue */
__device__ int gpu_queue_size(int *que){
    return que[QUEUE_SIZE+1] - que[QUEUE_SIZE];
}

/* Push a variable into the back of the queue */
__device__ void gpu_queue_push(int *que, int val){
    assert(gpu_queue_size(que) < QUEUE_SIZE);
    que[que[QUEUE_SIZE+1]%QUEUE_SIZE] = val;
    que[QUEUE_SIZE+1] += 1;
}

/* Pop a variable from the front of the queue */
__device__ void gpu_queue_pop(int *que){
    assert(!gpu_queue_empty(que));
    que[QUEUE_SIZE] += 1;
}

/* Pop k variables from the front of the queue */
__device__ void gpu_queue_pop_k(int *que, int k){
    assert(k <= gpu_queue_size(que));
    que[QUEUE_SIZE] += k;
}