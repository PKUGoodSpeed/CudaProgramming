#include <iostream>
#include "../ginkgo/GOrder.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>

using namespace std;


__global__ void test(){
    gpu_ginkgo::Order o(1024, 10, 180, 30);
    printf("\n\n===== Checking basic member functions =====\n\n");
    o.showOrderInfo();
    o.getAcked(150);
    printf("\nThe trade is acked!\n");
    o.showOrderInfo();
    o.qUpdateAgainstTrade(75, 15);
    o.showOrderInfo();
    o.qUpdateAgainstTrade(15, 15);
    o.showOrderInfo();
    int trade_size = 32;
    int dq = 0;
    int dqs = 0;
    int filled_qty = 0;
    int filled_pnl = 0;
    int bz = 0;
    o.filledAgainstTrade(trade_size, filled_qty, filled_pnl, dq, dqs, 1024, bz);
    o.showOrderInfo();
    printf("====Updating Info====\n");
    printf("dq = %d, dqs = %d, filled_qty = %d, filled_pnl = %d, trade_size = %d\n",
    dq, dqs, filled_qty, filled_pnl, trade_size);
    printf("\n\n===== Checking Q update =====\n\n");
    o.qUpdateAgainstBU(25);
    printf("Book Volume = 25\n");
    o.showOrderInfo();
    o.qUpdateAgainstBU(100);
    printf("\nBook Volume = 100\n");
    o.showOrderInfo();
    o.qUpdateAgainstBU(40);
    printf("\nBook Volume = 40\n");
    o.showOrderInfo();
    printf("\n\n===== Canceling this order =====\n\n");
    o.cancel(225. + 100);
    o.showOrderInfo();
}

int main(){
    def_dvec(float) dev_out(1, 0);
    test<<<1, 1>>>();
    return 0;
}
