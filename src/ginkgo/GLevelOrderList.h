#ifndef GLevelOrderList_h
#define GLevelOrderList_h
#include "../include/lglist.h"
#include "GOrder.h"
#include <cassert>

namespace gpu_ginkgo{
    /*
    Here we implement a Level Order List struct for GPU.
    This struct handles the orders in one price Level
    struct properties:
        level_price, level_qty
        cancel_time
        bool sell, cancel;
        list of orders: porders (pending orders), orders (acked orders)
    stuct member functions:
        Constructor,
        __device__ void procPendingOrders()
        __device__ void adjustQAgainstBU()
        __device__ void procTrade()
        __device__ void preCancel()
        __device__ int procCanceledOrders()
        __device__ void showLevelInfo()
        __device__ void showPendingOrderInfo()
        __device__ void showAckedOrderInfo()
    */
    template<const int LEVEL_ORDER_LIM = 5>
    struct LevelOrderList{
        /* For later convenience */
        typedef gpu_ginkgo::Order ggOrder;
        
        /* Level price and level total order quantity */
        int level_price, level_qty;
        
        /* Cancel Time */
        float cancel_time;
        
        /* sell or buy */
        bool sell;
        
        /* Bad Order needed to be canceled? */
        bool cancel;
        
        /* Pending Order List*/
        gpu_linearized_stl::list<ggOrder, LEVEL_ORDER_LIM> porders;
        
        /* Orders shown on the book */
        gpu_linearized_stl::list<ggOrder, LEVEL_ORDER_LIM> orders;
        
    public:
        /* Constructor */
        __device__ LevelOrderList(const int&p, bool s):
        level_price(p),
        level_qty(0),
        cancel_time(1.E9),
        sell(s),
        cancel(false) {}
        
        /* Move the Pending orders whose acked time is after the current time, onto 
        the book.
        Move first, then check fill/cancel status
        */
        __device__ void procPendingOrders(const int&bv, const float&t){
            for(auto j=porders.begin(); j!=porders.end(); ){
                if(porders.at(j).acked_time < t){
                    porders.at(j).getAcked(bv);
                    orders.push_back(porders.at(j));
                    j = porders.erase(j);
                }
                else porders.increment(j);
            }
        }
        
        /* 
        Addjusting Q value against Book update:
        this should happen only when the book volume is updated at the current price level
        and should be adjusted just after processing the pending order and canceling orders
        */
        __device__ void adjustQAgainstBU(const int& bz){
            for(auto j=orders.begin(); j!=orders.end(); orders.increment(j)){
                orders.at(j).qUpdateAgainstBU(bz);
            }
        }
        
        /* 
        Against Trade 
        Using the ggOrder struct, the function becomes more concise.
        */
        __device__ void procTrade(int &tv, int &fqty, int &fpnl, int &dq, int &dqs, int prc, int&bz){
            fqty = fpnl = dq = dqs = 0;
            for(auto j=orders.begin(); j!=orders.end(); ){
                orders.at(j).qUpdateAgainstTrade(dq, dqs);
                if(tv && orders.at(j).filledAgainstTrade(tv, fqty, fpnl, dq, dqs, prc, bz)){
                    j = orders.erase(j);
                }
                else orders.increment(j);
            }
            level_qty -= fqty;
        }
        
        /* We made a decision to cancel the orders at the current level */
        __device__ void preCancel(const float&ct){
            cancel = true;
            cancel_time = ct;
            for(auto j=orders.begin(); j!=orders.end(); orders.increment(j)){
                orders.at(j).cancel(cancel_time);
            }
            for(auto j=porders.begin(); j!=porders.end(); porders.increment(j)){
                porders.at(j).cancel(cancel_time);
            }
        }
        
        /* 
        Process canceled orders:
        When the time passed the cancel_time of an Order, the order is removed from the list
        return the total qty being removed.
        */
        __device__ int procCanceledOrders(const float&t){
            assert(cancel);
            int cqty = 0;
            for(auto j=orders.begin(); j!=orders.end(); ){
                if(orders.at(j).cancel_time < t){
                    level_qty -= orders.at(j).qty;
                    cqty += orders.at(j).qty;
                    j = orders.erase(j);
                }
                else orders.increment(j);
            }
            return cqty;
        }
        
        /* Sending New Order */
        __device__ void sendNewOrder(const int&q, const int&at){
            porders.push_back(ggOrder(level_price, q, at, level_qty));
            level_qty += q;
        }
        
        /* Showing Basic level information */
        __device__ void showLevelInfo(){
            printf("\n***** Showing Level Information *****\n");
            printf("level_price = %d, level_qty = %d, side = ", level_price, level_qty);
            if(sell) printf("SELL");
            else printf("BUY");
            printf(", To be canceled: %s\n", cancel?"TRUE":"FALSE");
            printf("*************************************\n");
        }
        
        /* Showing Pending Order List information */
        __device__ void showPendingOrderInfo(){
            printf("\n****** Showing Pending Orders ******\n");
            if(porders.empty()){
                printf("There is no pending orders\n");
            }
            else{
                for(auto j=porders.begin(); j!=porders.end(); porders.increment(j)){
                    porders.at(j).showOrderInfo();
                }
            }
            printf("************************************\n");
        }
        
        /* Showing Book Order List information */
        __device__ void showAckedOrderInfo(){
            printf("\n******* Showing Acked Orders *******\n");
            if(orders.empty()){
                printf("There is no acked orders\n");
            }
            else{
                for(auto j=orders.begin(); j!=orders.end(); orders.increment(j)){
                    orders.at(j).showOrderInfo();
                }
            }
            printf("************************************\n");
        }
    };
}


#endif