#ifndef lglist_h
#define lglist_h
#include "lgstack.h"
#include <cassert>

namespace gpu_linearized_stl{
    /*
    Here we use array to implement linked listed, which can be used on GPU.
    We define a node and iterator within the list struct.
    This list contains the following member functions:
        empty()
        size()
        max_size()
        full()
        increment(idx)
        decrement(idx)
        begin()
        end()
        front()
        back()
        push_front(val)
        push_back(val)
        pop_front()
        pop_back()
        insert(iterator, val)
        find(val)
        erase(iterator)
        reverse()
    So far the sort() function and mergeSort() function does not work properly.
    */
    template<typename T, const int MAX_SIZE = 126>
    struct list{
        /* Nodes on the list, we use integers to denote next and prev */
        T container[MAX_SIZE];
        int prev[MAX_SIZE+2], next[MAX_SIZE+2];
        gpu_linearized_stl::stack<int, MAX_SIZE> index_pool;
        int head, tail, length;
    public:
        __device__ list(): head(MAX_SIZE), tail(MAX_SIZE+1), length(0){
            memset(prev, -1, sizeof(prev));
            memset(next, -1, sizeof(next));
            next[MAX_SIZE] = MAX_SIZE + 1;
            prev[MAX_SIZE + 1] = MAX_SIZE;
            for(int i=0;i<MAX_SIZE;++i) index_pool.push(i);
        }
        __device__ bool empty(){
            return !length;
        }
        __device__ int size(){
            return length;
        }
        __device__ int max_size(){
            return MAX_SIZE;
        }
        __device__ bool full(){
            return index_pool.empty();
        }
        __device__ int increment(int &idx){
            assert(idx>=0 && idx<=MAX_SIZE+1);
            return idx = next[idx];
        }
        __device__ int decrement(int &idx){
            assert(idx>=0 && idx<=MAX_SIZE+1);
            return idx = prev[idx];
        }
        __device__ int begin(){
            return next[head];
        }
        __device__ int end(){
            return tail;
        }
        __device__ T front(){
            assert(length);
            return container[next[head]];
        }
        __device__ T back(){
            assert(length);
            return container[prev[tail]];
        }
        __device__ T& at(const int &where){
            assert(next[where]!=-1 && prev[where]!=-1);
            return container[where];
        }
        __device__ void push_front(const T&val){
            assert(length < MAX_SIZE);
            int idx = index_pool.top(), tmp = next[head];
            index_pool.pop();
            container[idx] = val;
            prev[idx] = head;
            next[idx] = tmp;
            prev[tmp] = idx;
            next[head] = idx;
            ++length;
        }
        __device__ void push_back(const T&val){
            assert(length < MAX_SIZE);
            int idx = index_pool.top(), tmp = prev[tail];
            index_pool.pop();
            container[idx]= val;
            prev[idx] = tmp;
            next[idx] = tail;
            prev[tail] = idx;
            next[tmp] = idx;
            ++length;
        }
        __device__ void pop_front(){
            assert(length);
            int idx = next[head];
            next[head] = next[idx];
            prev[next[idx]] = head;
            prev[idx] = next[idx] =-1;
            --length;
            index_pool.push(idx);
        }
        __device__ void pop_back(){
            assert(length);
            int idx = prev[tail];
            prev[tail] = prev[idx];
            next[prev[idx]] = tail;
            prev[idx] = next[idx] = -1;
            --length;
            index_pool.push(idx);
        }
        __device__ int insert(int where, const T&val){
            assert(length< MAX_SIZE);
            if(next[where]==-1 || prev[where]==-1){
                this->push_back(val);
                return prev[tail];
            }
            int idx = index_pool.top(), tmp = prev[where];
            index_pool.pop();
            container[idx] = val;
            prev[idx] = tmp;
            next[idx] = where;
            prev[where] = idx;
            next[tmp] = idx;
            ++length;
            return idx;
        }
        __device__ int find(const T&val){
            int p = next[head];
            for(;p!=tail && container[p]!=val; p = next[p]);
            return p;
        }
        __device__ int erase(int where){
            if(next[where]==-1 || prev[where]==-1) return this->begin();
            int tmp_prev = prev[where], tmp_next = next[where];
            prev[tmp_next] = tmp_prev;
            next[tmp_prev] = tmp_next;
            prev[where] = next[where] = -1;
            --length;
            index_pool.push(where);
            return tmp_next;
        }
        __device__ void reverse(){
            for(int p=head;p!=-1;p=prev[p]){
                int tmp = next[p];
                next[p] = prev[p];
                prev[p] = tmp;
            }
            int tmp = head;
            head = tail;
            tail = tmp;
        }
    };
}

#endif