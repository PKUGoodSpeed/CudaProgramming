#ifndef gpu_list_h
#define gpu_list_h
#include <cassert>

template<typename T>
struct gpu_list{
    /* Basic Nodes for the list */
    struct ListNode{
        T val;
        ListNode *prev, *next;
        __device__ ListNode(const T &x, ListNode *p = NULL, ListNode *n = NULL){
            val = x;
            prev = p;
            next = n;
        }
    };
    
    /* Iterator for the list */
    struct iterator{
        ListNode *ptr;
        __device__ iterator(ListNode *p = NULL){
            ptr = p;
        }
        __device__ iterator &operator =(const iterator &iter){
            ptr = iter.ptr;
            return *this;
        }
        __device__ void operator ()(const iterator &iter){
            ptr = iter.ptr;
        }
        __device__ void operator +=(const int &k){
            for(int i=0;i<k&&ptr;i++, ptr = ptr->next);
        }
        __device__ void operator -=(const int &k){
            for(int i=0;i<k&&ptr;i++, ptr = ptr->prev);
        }
        __device__ void operator ++(){
            this->operator +=(1);
        }
        __device__ void operator ++(int){
            ++(*this);
        }
        __device__ void operator --(){
            this->operator -=(1);
        }
        __device__ void operator --(int){
            --(*this);
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
        __device__ ListNode*& next(){
            return ptr->next;
        }
        __device__ ListNode*& prev(){
            return ptr->prev;
        }
        __device__ ~iterator(){
            delete ptr;
        }
    };
    
    iterator head, tail;
    int length;
public:
    /* Initialization */
    __device__ gpu_list(){
        head = tail = iterator(new ListNode(0));
        length = 0;
    }
    /* Empty */
    __device__ bool empty(){
        return !length;
    }
    /* Size */
    __device__ int size(){
        return length;
    }
    /* Begin */
    __device__ iterator& begin(){
        return head;
    }
    /* End */
    __device__ iterator& end(){
        return tail;
    }
    /* Front */
    __device__ T& front(){
        assert(!this->empty());
        return *head;
    }
    /* Back */
    __device__ T& back(){
        assert(!this->empty());
        return tail.prev()->val;
    }
    /* Push front */
    __device__ void push_front(const T& val){
        head.prev() = new ListNode(val, NULL, head.ptr);
        --head;
        ++length;
    }
    /* Push back */
    __device__ void push_back(const T&val){
        *tail = val;
        tail.next() = new ListNode(0, tail.ptr, NULL);
        ++tail;
        ++length;
    }
    /* Pop front */
    __device__ void pop_front(){
        assert(!this->empty());
        ++head;
        delete head.prev();
        head.prev() = NULL;
        --length;
    }
    /* Pop back */
    __device__ void pop_back(){
        assert(!this->empty());
        --tail;
        delete tail.next();
        tail.next() = NULL;
        --length;
    }
    /* insert */
    __device__ void insert(iterator& iter, const T &val){
        assert(iter.ptr);
        ListNode* tmp = new ListNode(val, iter.prev(), iter.ptr);
        iter.prev() = tmp;
        if(tmp->prev) tmp->prev->next = tmp;
        else --head;
        ++length;
        return;
    }
    /* erase */
    __device__ iterator& erase(iterator &iter){
        if(!iter.ptr || iter == this->end()) return iter;
        if(iter == this->begin()){
            this->pop_front();
            return head;
        }
        ListNode *p = iter.prev(), *n = iter.next();
        p->next = n;
        n->prev = p;
        delete iter.ptr;
        iter.ptr = n;
        --length;
        return iter;
    }
};

#endif