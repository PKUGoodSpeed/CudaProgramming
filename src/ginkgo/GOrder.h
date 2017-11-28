#ifndef GOrder_h
#define GOrder_h
#include <cassert>

namespace gpu_ginkgo{
    /*
    Here we implement an Order struct which can be used in GPUs.
    The Order struct contains the following properties:
        int price, qty, q_before, q_after, qs_before, qs_after;
        float acked_time, cancel_time;
    The Order struct must have the copy operators.
    The Order contains the following member functions:
        getAct()
        cancel()
        qUpdateAgainstTrade()
        qUpdateAgainstBU()
        filledAgainstTrade()
        needCancel()
        needSwipe()
        showOrderInfo() print Order information
    */
    struct Order{
        // Basic information
        int price, qty;
        
        // Position in the Queue, book contribution
        int q_before, q_after;
        
        // Position in the Queue, self contribution, only consider trades before us
        int q_self;
        
        // Acked time and cancel time
        float acked_time, cancel_time;
        
    public:
        /*
        We will make sure the order is always initialized in sending new Order processes,
        for which the parameters passed in are price, qty, acked time, and sell qty for 
        the corresponding price.
        */
        
        /* Default constructor */
        __device__ Order():
        price(0), 
        qty(0),
        q_before(0),
        q_after(0),
        q_self(0),
        acked_time(0),
        cancel_time(0.) {}
        
        /* Constructor */
        __device__ Order(const int&p, const int&q, const float&at, const int &qs):
        price(p), 
        qty(q),
        q_before(0),
        q_after(0),
        q_self(qs),
        acked_time(at),
        cancel_time(-1.) {}
        
        /* Copy constructor */
        __device__ Order(const Order&o):
        price(o.price),
        qty(o.qty),
        q_before(o.q_before),
        q_after(o.q_after),
        q_self(o.q_self),
        acked_time(o.acked_time),
        cancel_time(o.cancel_time) {}
        
        /* Copy operator */
        __device__ Order& operator =(const Order& o){
            price = o.price;
            qty = o.qty;
            q_before = o.q_before;
            q_after = o.q_after;
            q_self = o.q_self;
            acked_time = o.acked_time;
            cancel_time = o.cancel_time;
            return *this;
        }
        
        /* Getting Acked (This must be called after immediate filling adjust to the book)*/
        __device__ void getAcked(const int& book_size, const int&aqty = -1){
            q_before = book_size;
            if(aqty >= 0) q_self = aqty;
        }
        
        /* Canceling */
        __device__ void cancel(const float& ct){
            assert(cancel_time < 0.);
            cancel_time = ct;
        }
        
        /* Updating Q values against a particular trade. (This must be called after immediate filling adjust to the book) 
        During the previous trade update, we need to tract the dq and dqs information for 
        a particular price level.
        */
        __device__ void qUpdateAgainstTrade(const int&dq, const int&dqs){
            assert(q_before >= dq);
            assert(q_self >= dqs);
            q_before -= dq;
            q_self -= dqs;
        }
        
        /* Filled agaist trade 
        If this order is totally filled, return true;
        otherwise, return false;
        */
        __device__ bool filledAgainstTrade(int &tv, int &filled_qty, int&filled_pnl, int &dq, int &dqs, int prc, int &bz){
            if(!tv) return false;
            int delta_qty = min(q_before, tv);
            tv -= delta_qty;
            q_before -= delta_qty;
            dq += delta_qty;
            bz -= delta_qty;
            if(!tv) return false;
            delta_qty = min(qty, tv);
            tv -= delta_qty;
            qty -= delta_qty;
            dqs += delta_qty;
            filled_qty += delta_qty;
            filled_pnl += delta_qty * prc;
            printf("An order is filled against a aggressive trade:filled_qty = %d, qty_left = %d; \n", delta_qty, qty);
            return !qty;
        }
        
        /* Updating Q values against a book update. (This must be called after immediate filling adjust to the book) 
        During the previous trade update, we need to tract the dq and dqs information for 
        a particular price level.
        */
        __device__ void qUpdateAgainstBU(const int &bz){
            int total_q = q_before + q_after;
            if(bz == total_q) return;
            if(bz > total_q){
                q_after += bz - total_q;
                return;
            }
            q_before = bz * q_before / total_q;
            q_after = bz - q_before;
        }
        
        /* Checking whether need to be swiped*/
        __device__ bool needSwiped(const bool&sell, const int&ap, const int&bp){
            return (sell && price <= bp) || (!sell && price >= ap);
        }
        
        /* Checking whether the order needs to be canceled or not */
        __device__ bool needCancel(const int&pmin, const int&pmax){
            return price > pmax || price < pmin;
        }
        
        /* Show Order Info */
        __device__ void showOrderInfo(){
            printf("From block #%d and thread #%d: \n", blockIdx.x, threadIdx.x);
            printf("Getting Order Info: \nprice: %d, qty: %d, q_befor: %d, q_after: %d, q_self: %d, acked_time: %.2f, cancel_time %.2f\n", 
            price, qty, q_before, q_after, q_self, acked_time, cancel_time);
        }
        
    };
}

#endif