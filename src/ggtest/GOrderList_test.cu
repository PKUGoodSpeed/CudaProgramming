#include <iostream>
#include "../ginkgo/GOrderList.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

__global__ void test(){
    int pos = 0, ppos = 0, pnl = 0, tqty;
    // Creating an OrderList struct
    gpu_ginkgo::OrderList<100, 6> ggol(true, 1024, 10);
    ggol.getTime(1.5,  1.0);
    printf("<<< CREATING A NEW ORDER LIST STRUCTURE >>>\n");
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Sending new Orders
    printf("<<< SENDING FOUR NEW ORDERS >>>\n");
    int q_lim = 25;
    ggol.sendNewOrder(1024, q_lim);
    ggol.sendNewOrder(1024, q_lim);
    ggol.sendNewOrder(1025, q_lim);
    ggol.sendNewOrder(1026, q_lim);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Sending new Orders
    printf("<<< SENDING ANOTHER ORDERS >>>\n");
    q_lim = 20;
    ggol.getTime(1.7, 1.0);
    ggol.sendNewOrder(1026, q_lim);
    ggol.sendNewOrder(1027, q_lim);
    ggol.sendNewOrder(1028, q_lim);
    ggol.sendNewOrder(1029, q_lim);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Sending multiple orders to the last_level
    printf("<<< SENDING ANOTHER ORDERS >>>\n");
    ggol.getTime(2.0, 1.0);
    q_lim = 2;
    ggol.sendNewOrder(1029, q_lim);
    q_lim = 2;
    ggol.sendNewOrder(1029, q_lim);
    q_lim = 2;
    ggol.sendNewOrder(1029, q_lim);
    q_lim = 2;
    ggol.sendNewOrder(1029, q_lim);
    q_lim = 2;
    ggol.sendNewOrder(1029, q_lim);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Acking new orders
    ggol.getTime(2.6, 1.0);
    printf("<<< ACKING NEW ORDERS >>>");
    for(auto j=ggol.porders.begin(); j!=ggol.porders.end(); ){
        if(ggol.porders.at(j).acked_time < ggol.cur_time){
            ggol.ackPendingOrder(j, 10);
        }
        else ggol.porders.increment(j);
    }
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Canceling orders with price = 1026
    printf("<<< CANCELING ORDERS WITH PRICE = 1026 >>>");
    for(auto j=ggol.porders.begin(); j!=ggol.porders.end(); ){
        if(ggol.porders.at(j).price == 1026){
            ggol.cancelPendingOrder(j);
        }
        else ggol.porders.increment(j);
    }
    
    for(auto j=ggol.orders.begin(); j!=ggol.orders.end(); ){
        if(ggol.orders.at(j).price == 1026){
            ggol.cancelAckedOrder(j);
        }
        else ggol.orders.increment(j);
    }
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Swipe orders with price = 1024
    printf("<<< SWIPING ORDERS WITH PRICE = 1024 >>>");
    for(auto j=ggol.orders.begin(); j!=ggol.orders.end(); ){
        if(ggol.orders.at(j).price == 1024){
            ggol.swipeAckedOrder(j);
        }
        else ggol.orders.increment(j);
    }
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Acking new orders
    ggol.getTime(2.71, 1.0);
    printf("<<< ACKING NEW ORDERS >>>");
    for(auto j=ggol.porders.begin(); j!=ggol.porders.end(); ){
        if(ggol.porders.at(j).acked_time < ggol.cur_time){
            ggol.ackPendingOrder(j, 10);
        }
        else ggol.porders.increment(j);
    }
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // A trade comes with price = 1026, qty = 17
    printf("<<< A AGGRESSIVE TRADE COMES WITH PRICE = 1026, QTY = 17 >>>");
    int tv = 17, prc = 1026;
    int book_size[100];
    ggol.getTradedThrough(tv, prc, book_size);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Canceling orders with price = 1029
    ggol.getTime(2.71, 0.1);
    printf("<<< CANCELING ORDERS WITH PRICE = 1029 >>>");
    for(auto j=ggol.porders.begin(); j!=ggol.porders.end(); ){
        if(ggol.porders.at(j).price == 1029){
            ggol.cancelPendingOrder(j);
        }
        else ggol.porders.increment(j);
    }
    
    for(auto j=ggol.orders.begin(); j!=ggol.orders.end(); ){
        if(ggol.orders.at(j).price == 1029){
            ggol.cancelAckedOrder(j);
        }
        else ggol.orders.increment(j);
    }
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Canceling orders with price = 1029
    ggol.getTime(3.91, 0.1);
    printf("<<< CLEAN UP CANCEL ORDERS >>>");
    for(auto j=ggol.corders.begin(); j!= ggol.corders.end(); ){
        if(ggol.corders.at(j).cancel_time < ggol.cur_time){
            ggol.cancelOrder(j);
        }
        else ggol.corders.increment(j);
    }
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.reset(pos, pnl, tqty);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // Test finished
    printf("\n <<< TEST FINISHED !!! >>>\n");
}

int main(){
    def_dvec(float) dev_out(1, 0);
    test<<<1, 1>>>();
    return 0;
}