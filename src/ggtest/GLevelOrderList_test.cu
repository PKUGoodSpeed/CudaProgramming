#include <iostream>
#include "../ginkgo/GLevelOrderList.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>

using namespace std;
typedef gpu_ginkgo::LevelOrderList<5> gglol;

__global__ void test(){
    gglol *p;
    p = new gglol(1024, true);
    printf("p = new gglol(1024);");
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    printf("\n\n<<< SENDING A NEW ORDER >>>");
    p->sendNewOrder(20, 3.0);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    printf("\n\n<<< SENDING A NEW ORDER >>>");
    p->sendNewOrder(20, 3.0);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    printf("\n\n<<< SENDING A NEW ORDER >>>");
    p->sendNewOrder(20, 4.0);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    printf("\n\n<<< SOME ORDER BEING ACKED >>>");
    p->procPendingOrders(20, 3.1);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    printf("\n\n<<< CANCELING THESE ORDERS >>>");
    p->preCancel(5.2);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    printf("\n\n<<< SOME ORDER BEING ACKED >>>");
    p->procPendingOrders(20, 4.1);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    printf("\n\n<<< SOME TRADE COMING >>>\n");
    int tv = 55, fqty = 0, fpnl = 0, dq = 0, dqs = 0, bz = 0;
    printf("Trade size = %d, filled qty = %d, filled pnl = %d", tv, fqty, fpnl);
    p->procTrade(tv, fqty, fpnl, dq, dqs, 1024, bz);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    printf("Trade size = %d, filled qty = %d, filled pnl = %d\n\n", tv, fqty, fpnl);
    
    printf("\n\n<<< BOOK UPDATE OCCURS >>>");
    p->adjustQAgainstBU(100);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    printf("\n\n<<< BOOK UPDATE OCCURS >>>");
    p->adjustQAgainstBU(20);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    printf("\n\n<<< CANCEL TIME COMES >>>");
    p->procCanceledOrders(5.5);
    p->showLevelInfo();
    p->showPendingOrderInfo();
    p->showAckedOrderInfo();
    //
    
    delete p;
    p = NULL;
    if(!p) printf("\n\n SUCCESSFUL REMOVE THE LEVEL\n\n");
}

int main(){
    def_dvec(float) dev_out(1, 0);
    test<<<1, 1>>>();
    return 0;
}