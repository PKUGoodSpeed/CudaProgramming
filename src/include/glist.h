#ifndef glist_h
#define glist_h
#include <cassert>

namespace gpu_stl{
    /*
    Here I implement a gpu version of stl linked list using pointers,
    which contains:
        struct ListNode
        struct iterator
    and member functions:
        empty()
        size()
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
        clear()
    So far the sort() function and mergeSort() function does not work properly.
    */
    template<typename T>
    struct list{
        /* Nodes on the list, we use double direction list */
        struct ListNode{
            T val;
            ListNode *prev, *next;
            __device__ ListNode(const T&v, ListNode *p = NULL, ListNode *n = NULL){
                val = v;
                prev = p;
                next = n;
            }
        };
        
        /* Iterator for the list */
        struct iterator{
            /*
            The iterator support all stl operators
            */
            ListNode *ptr;
            __device__ iterator(ListNode *p = NULL): ptr(p) {}
            __device__ iterator(const iterator& iter): ptr(iter.ptr) {}
            __device__ iterator& operator =(const iterator &iter){
                ptr = iter.ptr;
                return *this;
            }
            __device__ void operator ()(const iterator &iter){
                ptr = iter.ptr;
            }
            __device__ iterator& operator +=(const int &k){
                for(int i=0;i<k&&ptr;i++, ptr = ptr->next);
                return *this;
            }
            __device__ iterator& operator -=(const int &k){
                for(int i=0;i<k&&ptr;i++, ptr = ptr->prev);
                return *this;
            }
            __device__ iterator& operator ++(){
                this->operator +=(1);
                return *this;
            }
            __device__ iterator& operator ++(int){
                this->operator +=(1);
                return *this;
            }
            __device__ iterator& operator --(){
                this->operator -=(1);
                return *this;
            }
            __device__ iterator& operator --(int){
                this->operator -=(1);
                return *this;
            }
            __device__ bool operator ==(const iterator &iter){
                return ptr == iter.ptr;
            }
            __device__ bool operator !=(const iterator &iter){
                return ptr != iter.ptr;
            }
            __device__ T& operator *(){
                return this->ptr->val;
            }
        };
        
        iterator head, tail;
        int length;
    public:
        __device__ list(){
            head = tail = iterator(new ListNode(0));
            length = 0;
        }
        __device__ bool empty(){
            return !length;
        }
        __device__ int size(){
            return length;
        }
        __device__ iterator begin(){
            return head;
        }
        __device__ iterator end(){
            return tail;
        }
        __device__ T front(){
            assert(!this->empty());
            return head.ptr->val;
        }
        __device__ T back(){
            assert(!this->empty());
            return tail.ptr->prev->val;
        }
        __device__ void push_front(const T&val){
            head.ptr->prev = new ListNode(val, NULL, head.ptr);
            --head;
            ++length;
        }
        __device__ void push_back(const T&val){
            tail.ptr->val = val;
            tail.ptr->next = new ListNode(0, tail.ptr, NULL);
            ++tail;
            ++length;
        }
        __device__ void pop_front(){
            assert(!this->empty());
            ++head;
            delete head.ptr->prev;
            head.ptr->prev = NULL;
            --length;
        }
        __device__ void pop_back(){
            assert(!this->empty());
            --tail;
            delete tail.ptr->next;
            tail.ptr->next = NULL;
            --length;
        }
        __device__ iterator insert(iterator iter, const T&val){
            if(!iter.ptr || iter == tail){
                this->push_back(val);
                return iterator(tail.ptr->prev);
            }
            if(iter == head){
                this->push_front(val);
                return head;
            }
            ListNode *tmp = new ListNode(val, iter.ptr->prev, iter.ptr);
            tmp->prev->next = tmp;
            tmp->next->prev = tmp;
            ++length;
            return iterator(tmp);
        }
        __device__ iterator erase(iterator iter){
            if(!iter.ptr || iter == tail) return iter;
            if(iter == head){
                this->pop_front();
                return head;
            }
            iter.ptr->prev->next = iter.ptr->next;
            iter.ptr->next->prev = iter.ptr->prev;
            ListNode *n = iter.ptr->next;
            delete iter.ptr;
            return iterator(n);
        }
        __device__ void reverse(){
            ListNode *left = head.ptr, *right = tail.ptr->prev;
            for(int i=0;i<length/2;++i){
                T tmp_val = left->val;
                left->val = right->val;
                right->val = tmp_val;
                left = left->next;
                right = right->prev;
            }
        }
        __device__ iterator find(const T&val){
            iterator p;
            for(p=this->begin();p!=this->end()&&p.ptr->val!=val;++p);
            return p;
        }
        __device__ void clear(){
            for(auto p=head.ptr->next;p;p=p->next) delete p->prev;
            head = tail;
            length = 0;
        }
        __device__ ~list(){
            for(auto p=head.ptr->next;p;p=p->next) delete p->prev;
            delete tail.ptr;
        }
    };
}

#endif