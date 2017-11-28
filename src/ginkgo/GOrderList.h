#ifndef GOrderList_h
#define GOrderList_h
#include "../include/lglist.h"
#include "GOrder.h"
#include <cassert>

namespace gpu_ginkgo{
    /*
    Here we implement an Order List struct for GPU,
    This Order List contains the Orders from different levels.
    It combines three types of list:
        Pending orders
        Acked orders
        canceled orders
    It contains the following member functions:
        constructor
        __device__ void getTime(const float&t, const float&dt)
        __device__ void cancelOrder(int &j)
        __device__ void swipeCancelOrder(int &j)
        __device__ void swipeAckedOrder(int &j)
        __device__ void cancelPendingOrder(int &j)
        __device__ void cancelAckedOrder(int &j)
        __device__ void ackPendingOrder(int &j, const int&bz)
        __device__ void getTradedThrough(int &tv, int prc)
        __device__ void sendNewOrder(const int&prc, int &q_lim)
        __device__ void updatePendingOrders(int *book_sizes)
        __device__ void updateCancelOrders(const int&prc_line, int *book_sizes)
        __device__ void updateAckedOrders(const int&prc_line, int *book_sizes)
        __device__ void updatePositions(int &ppos, int &pos)
        __device__ void updatePnl(int &pnl)
        __device__ void preCanceling(const int&lp, const int&hp)
        __device__ void sendNewSellingOrders(const int&abbo, const int&bbo_level, int &q_lim)
        __device__ void sendNewBuyingOrders(const int&bbbo, const int&bbo_level, int &q_lim)
        __device__ void reset(int &ppos, int &pos, int &pnl)
        __device__ void showLevelQtyInfo()
        __device__ void showPendingOrderInfo()
        __device__ void showAckedOrderInfo()
        __device__ void showCanceledOrderInfo()
        __device__ void showUpdateInfo()
    */
    template<const int ORDER_LIM = 100, const int LEVEL_LIM = 200>
    struct OrderList{
        /* For later convenience */
        typedef gpu_ginkgo::Order ggOrder;
        
        /* Pending Order List */
        gpu_linearized_stl::list<ggOrder, ORDER_LIM> porders;
        
        /* Acked Orders that are shown on the book, and not gonna canceled */
        gpu_linearized_stl::list<ggOrder, ORDER_LIM> orders;
        
        /* Acked Orders that are determined to be canceled but are still on the book */
        gpu_linearized_stl::list<ggOrder, ORDER_LIM> corders;
        
        /* Buy or Sell */
        bool sell;
        
        /* Total qty and total pending qty */
        int side_qty, delta_fqty, delta_pnl;
        
        /* base price */
        int base_price;
        
        /* Array to record qty and filled qtys */
        int qty[LEVEL_LIM];   // Current all qtys in a particular price level
        int acked_qty[LEVEL_LIM]; //Current acked_qtys in a particular price level
        
        /* max Level qty */
        int max_level_qty;
        
        /* Current time and latency */
        float cur_time, ltcy;
        
    public:
        /* Constructor */
        __device__ OrderList(const bool&s, const int&base_p, const int&mlq):
        sell(s),
        side_qty(0),
        delta_fqty(0),
        delta_pnl(0),
        base_price(base_p),
        max_level_qty(mlq),
        cur_time(0),
        ltcy(0) {
            memset(qty, 0, sizeof(qty));
            memset(acked_qty, 0, sizeof(acked_qty));
        }
        
        /* Getting time information */
        __device__ void getTime(const float&t, const float&dt){
            cur_time = t;
            ltcy = dt;
        }
        
        /* Check whether a price level is empty or not */
        __device__ bool levelEmpty(const int&prc){
            if(prc < base_price || prc >= base_price + LEVEL_LIM) return false;
            return qty[prc - base_price] == 0;
        }
        
        /* Get Level index from outside */
        __device__ int getLevel(const int&prc){
            return prc - base_price;
        }
        
        /* Cancel Order */
        __device__ void cancelOrder(int &j){
            int prc = corders.at(j).price, tmp_qty = corders.at(j).qty;
            qty[prc - base_price] += tmp_qty;
            acked_qty[prc - base_price] -= tmp_qty;
            side_qty -= tmp_qty;
            j = corders.erase(j);
        }
        
        /* Swipe Cancel Order */
        __device__ void swipeCancelOrder(int &j){
            int prc = corders.at(j).price, tmp_qty = corders.at(j).qty;
            qty[prc - base_price] += tmp_qty;
            side_qty -= tmp_qty;
            acked_qty[prc - base_price] -= tmp_qty;
            delta_fqty += tmp_qty;
            delta_pnl += tmp_qty * prc;
            j = corders.erase(j);
        }
        
        /* Swipe Acked Order */
        __device__ void swipeAckedOrder(int &j){
            int prc = orders.at(j).price, tmp_qty = orders.at(j).qty;
            qty[prc - base_price] -= tmp_qty;
            side_qty -= tmp_qty;
            acked_qty[prc - base_price] -= tmp_qty;
            delta_fqty += tmp_qty;
            delta_pnl += tmp_qty * prc;
            j = orders.erase(j);
        }
        
        /* Cancel Pending Order */
        __device__ void cancelPendingOrder(int &j){
            if(qty[porders.at(j).price - base_price] > 0){
                qty[porders.at(j).price - base_price] *= -1;
            }
            if(porders.at(j).acked_time < cur_time + ltcy){
                porders.at(j).cancel(cur_time + ltcy);
                porders.increment(j);
            }
            else{
                qty[porders.at(j).price - base_price] += porders.at(j).qty;
                side_qty -= porders.at(j).qty;
                j = porders.erase(j);
            }
        }
        
        /* Cancel Acked Order */
        __device__ void cancelAckedOrder(int &j){
            if(qty[orders.at(j).price - base_price] > 0){
                qty[orders.at(j).price - base_price] *= -1;
            }
            orders.at(j).cancel(cur_time + ltcy);
            corders.push_back(orders.at(j));
            j = orders.erase(j);
        }
        
        /* Ack Pending orders */
        __device__ void ackPendingOrder(int &j, const int&bz){
            assert(porders.at(j).acked_time < cur_time && porders.at(j).qty != 0);
            int prc = porders.at(j).price;
            porders.at(j).getAcked(bz, acked_qty[prc - base_price]);
            acked_qty[prc - base_price] += porders.at(j).qty;
            if(qty[prc - base_price] > 0){
                orders.push_back(porders.at(j));
            }
            else{
                corders.push_back(porders.at(j));
            }
            j = porders.erase(j);
        }
        
        /* Trade Through */
        __device__ void getTradedThrough(int &tv, int prc, int *bz){
            if(!qty[prc - base_price]) return;
            int fq = 0, fpnl = 0, dq =0, dqs = 0;
            fq = fpnl = dq = dqs = 0;
            if(qty[prc - base_price] > 0){
                /* Loop through the acked order list */
                for(auto j=orders.begin(); j!=orders.end(); ){
                    if(orders.at(j).price != prc) {
                        orders.increment(j);
                        continue;
                    }
                    orders.at(j).qUpdateAgainstTrade(dq, dqs);
                    if(tv && orders.at(j).filledAgainstTrade(tv, fq, fpnl, dq, dqs, prc, bz[prc - base_price])){
                        j = orders.erase(j);
                    }
                    else orders.increment(j);
                }
                qty[prc - base_price] -= fq;
                acked_qty[prc - base_price] -= fq;
            }
            else{
                /* Loop through the canceled order list */
                for(auto j=corders.begin();j!=corders.end(); ){
                    if(corders.at(j).price != prc){
                        corders.increment(j);
                        continue;
                    }
                    corders.at(j).qUpdateAgainstTrade(dq, dqs);
                    if(tv && corders.at(j).filledAgainstTrade(tv, fq, fpnl, dq, dqs, prc, bz[prc - base_price])){
                        j = corders.erase(j);
                    }
                    else corders.increment(j);
                }
                qty[prc - base_price] += fq;
                acked_qty[prc - base_price] -= fq;
            }
            side_qty -= fq;
            delta_fqty += fq;
            delta_pnl += fq * prc;
        }
        
        /* Sending new orders */
        __device__ void sendNewOrder(const int&prc, int &q_lim){
            assert(prc >= base_price && prc < base_price + LEVEL_LIM);
            if(!q_lim || qty[prc-base_price] < 0 || qty[prc-base_price] == max_level_qty) return;
            int q = min(q_lim, max_level_qty - qty[prc-base_price]);
            porders.push_back(ggOrder(prc, q, cur_time+ltcy, qty[prc-base_price]));
            qty[prc-base_price] += q;
            q_lim -= q;
            side_qty += q;
        }
        
        /* 
        For each Loop at a particular book update, we need to first determine the current 
        status, after that then determine the next actions.
        Following is the things that we need do one by one:
            1. Update pending order list:
                (1) put the acted order on to the order list or cancel order list
            2. Update canceling order list:
                (1) determine which orders is actually canceled at this point
                (2) determine which orders are swiped by the price move
                (3) update the q_before and q_after values
            3. Update the acked order list:
                (1) determine which orders are swiped by the price move
                (2) update the q_before and q_after values
            4. Reset and updating position, pending position and bbos
            5. Compute ask and bid bbos
            6. Loop through the pending order list and determine which orders need to be canceled
            7. Loop through the acked order list and determine which orders need to be canceled
            8. Send new orders due to ask, bid bbos and current pending position
                (1) update buy/sell new order status
                (2) update pending position
                (3) update sell/buy new order status
                (4) update pending position again
        */
        
        /* 
        1. Update pending order list 
        int *book_sizes is an array storing the current book size values.
        */
        __device__ void updatePendingOrders(int *book_sizes){
            for(auto j = porders.begin(); j!=porders.end(); ){
                if(porders.at(j).acked_time < cur_time){
                    /* book_sizes[porders.at(j).price - base_price] is the book size at the current level */
                    this->ackPendingOrder(j, book_sizes[porders.at(j).price - base_price]);
                }
                else porders.increment(j);
            }
        }
        
        /*
        2. Update canceling order list
            (1) determine which orders is actually canceled at this point
            (2) determine which orders are swiped by the price move
            (3) update the q_before and q_after values
        */
        __device__ void updateCancelOrders(const int&prc_line, int *book_sizes){
            for(auto j = corders.begin(); j!=corders.end(); ){
                int prc = corders.at(j).price;
                if(corders.at(j).cancel_time < cur_time){
                    // Time to cancel
                    this->cancelOrder(j);
                }
                else if((sell && prc<=prc_line) || (!sell && prc>=prc_line)){
                    // Swiped by the price move
                    this->swipeCancelOrder(j);
                }
                else{
                    // Update Q values
                    corders.at(j).qUpdateAgainstBU(book_sizes[prc - base_price]);
                    corders.increment(j);
                }
            }
        }
        
        /*
        3. Update the acked order list:
            (1) determine which orders are swiped by the price move
            (2) update the q_before and q_after values
        */
        __device__ void updateAckedOrders(const int&prc_line, int *book_sizes){
            for(auto j = orders.begin(); j!=orders.end(); ){
                int prc = orders.at(j).price;
                if((sell && prc<=prc_line) || (!sell && prc>=prc_line)){
                    this->swipeAckedOrder(j);
                }
                else{
                    orders.at(j).qUpdateAgainstBU(book_sizes[prc - base_price]);
                    orders.increment(j);
                }
            }
        }
        
        /*
        4. Update position and pending position
        */
        __device__ void updatePositions(int &pos){
            pos += (sell? -1:1) * delta_fqty;
            delta_fqty = 0;
        }
        
        /*
        5. Update pnl
        */
        __device__ void updatePnl(int &pnl){
            pnl += (sell? 1:-1) * delta_pnl;
            delta_pnl = 0;
        }
        
        /*
        6. Loop through the pending order list and determine which orders need to be canceled
        7. Loop through the acked order list and determine which orders need to be canceled
        */
        __device__ void preCanceling(const int&lp, const int&hp){
            // Cancel pending orders
            for(auto j=porders.begin(); j!=porders.end(); ){
                if(porders.at(j).price < lp || porders.at(j).price > hp){
                    this->cancelPendingOrder(j);
                }
                else porders.increment(j);
            }
            // Cancel acked orders
            for(auto j=orders.begin(); j!=orders.end(); ){
                if(orders.at(j).price < lp || orders.at(j).price > hp){
                    this->cancelAckedOrder(j);
                }
                else orders.increment(j);
            }
            
        }
        
        /*
        8. (1) Sending New selling orders
        BBO level is modified here.
        */
        __device__ void sendNewSellingOrders(const int&lp, const int&hp, int &q_lim){
            for(int prc = max(lp, base_price); prc < min(hp + 1, LEVEL_LIM+base_price) && q_lim; ++prc){
                this->sendNewOrder(prc, q_lim);
            }
        }
        
        /*
        8. (2) Sending New buying orders
        BBO level is modified here.
        */
        __device__ void sendNewBuyingOrders(const int&lp, const int&hp, int &q_lim){
            for(int prc = min(hp, base_price+LEVEL_LIM-1); prc >= max(lp, base_price) && q_lim; --prc){
                this->sendNewOrder(prc, q_lim);
            }
        }
        
        
        /* Resetting */
        __device__ void reset(int &pos, int &pnl, int &tqty){
            pos += (sell? -1:1) * delta_fqty;
            pnl += (sell? 1:-1) * delta_pnl;
            tqty += delta_fqty;
            
            delta_fqty = 0;
            delta_pnl = 0;
        }
        
        /* Show Basic information */
        __device__ void showLevelQtyInfo(){
            printf("\n***** Showing LevelQty Information *****\n");
            if(sell) printf("SELL\n");
            else printf("BUY\n");
            printf("Total qty: %d\n", side_qty);
            printf("price:\tlevel qty:\tlevel acked qty\n");
            for(int i=0; i<LEVEL_LIM; ++i) printf("%d :\t%d :\t%d\n", i+base_price, qty[i], acked_qty[i]);
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
        
        /* Showing Canceled Order List information */
        __device__ void showCanceledOrderInfo(){
            printf("\n****** Showing Canceled Orders ******\n");
            if(corders.empty()){
                printf("There is no canceled orders\n");
            }
            else{
                for(auto j=corders.begin(); j!=corders.end(); corders.increment(j)){
                    corders.at(j).showOrderInfo();
                }
            }
            printf("************************************\n");
        }
        
        
        
        /* Showing the Update Information */
        __device__ void showUpdateInfo(){
            printf("\n***** Showing Update Information *****\n");
            printf("Total filled qty = %d, filled pnl = %d\n", delta_fqty, delta_pnl);
            printf("*************************************\n");
        }
    };
}

#endif