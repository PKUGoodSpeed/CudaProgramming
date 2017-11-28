#include <iostream>
#include "../ginkgo/GOrderList.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

__global__ void test(){
    int pos = 0, ppos = 0, pnl = 0;
    // Creating an OrderList struct
    gpu_ginkgo::OrderList<100, 10> ggol(true, 1024, 10);
    ggol.getTime(1.5,  1.0);
    printf("<<< CREATING A NEW ORDER LIST STRUCTURE >>>\n");
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    int q_lim;
    // SENDING SELLING ORDERS
    printf("<<< SENDING NEW SELLING ORDERS >>>\n");
    q_lim = 2;
    ggol.sendNewSellingOrders(1029, 1033, q_lim);
    ggol.getTime(1.6, 1.0);
    q_lim = 2;
    ggol.sendNewSellingOrders(1029, 1033, q_lim);
    ggol.getTime(1.7, 1.0);
    q_lim = 2;
    ggol.sendNewSellingOrders(1030, 1034, q_lim);
    ggol.getTime(1.8, 1.0);
    q_lim = 2;
    ggol.sendNewSellingOrders(1031, 1035, q_lim);
    ggol.getTime(1.9, 1.0);
    q_lim = 37;
    ggol.sendNewSellingOrders(1029, 1033, q_lim);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.updatePositions(pos);
    ggol.updatePnl(pnl);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // SENDING NEW BUYING ORDERS
    printf("<<< SENDING NEW BUYING ORDERS >>>\n");
    q_lim = 2;
    ggol.sendNewBuyingOrders(1028, 1032, q_lim);
    ggol.getTime(2.0, 1.0);
    q_lim = 3;
    ggol.sendNewBuyingOrders(1027, 1031, q_lim);
    ggol.getTime(2.1, 1.0);
    q_lim = 4;
    ggol.sendNewBuyingOrders(1026, 1030, q_lim);
    ggol.getTime(2.2, 1.0);
    q_lim = 17;
    ggol.sendNewBuyingOrders(1028, 1032, q_lim);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.updatePositions(pos);
    ggol.updatePnl(pnl);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // 2.9s Update
    printf("<<< 2.9s BOOK UPDATES >>>\n");
    ggol.getTime(2.9, 0.17);
    int bz[10] = {10, 10, 15, 15, 20, 20, 15, 5, 25, 45};
    ggol.updatePendingOrders(bz);
    ggol.updateCancelOrders(1029, bz);
    ggol.updateAckedOrders(1029, bz);
    ggol.preCanceling(1030, 1032);
    q_lim = 37;
    ggol.sendNewSellingOrders(1029, 33, q_lim);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.updatePositions(pos);
    ggol.updatePnl(pnl);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // 3.2s Update
    printf("<<< 3.2s BOOK UPDATES >>>\n");
    ggol.getTime(3.2, 0.17);
    for(int i=0;i<10;++i) bz[i] = (i+1)*10;
    ggol.updatePendingOrders(bz);
    ggol.updateCancelOrders(1029, bz);
    ggol.updateAckedOrders(1029, bz);
    ggol.preCanceling(1030, 1032);
    q_lim = 37;
    ggol.sendNewSellingOrders(1029, 1033, q_lim);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.updatePositions(pos);
    ggol.updatePnl(pnl);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    printf("--------------------------------------------------------\n\n");
    
    // A trade comes with price = 1030, qty = 90;
    printf("<<< TRADE: PRICE = 1030, QTY = 90 >>>\n");
    ggol.getTime(3.3, 0.17);
    int tv = 75;
    ggol.getTradedThrough(tv, 1030, bz);
    ggol.showLevelQtyInfo();
    ggol.showPendingOrderInfo();
    ggol.showAckedOrderInfo();
    ggol.showCanceledOrderInfo();
    ggol.showUpdateInfo();
    ggol.updatePositions(pos);
    ggol.updatePnl(pnl);
    printf("position = %d, pending position = %d, pnl = %d \n", pos, ppos, pnl);
    for(int i=0;i<10;++i) printf("%d:  \t%d\n",1024+i, bz[i]);
    printf("\n");
    printf("--------------------------------------------------------\n\n");
    
    
    // Test finished
    printf("\n <<< TEST FINISHED !!! >>>\n");
}

int main(){
    def_dvec(float) dev_out(1, 0);
    test<<<1, 1>>>();
    return 0;
}