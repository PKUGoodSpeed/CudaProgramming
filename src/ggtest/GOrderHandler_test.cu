#include <iostream>
#include "../ginkgo/GOrderHandler.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

__global__ void test(){
    // Creating an OrderHandler struct
    gpu_ginkgo::OrderHandler<100, 10> ggoh(1024, 10);
    ggoh.showOrderBookInfo();
    ggoh.loadStrategy(33, 0., 0.);
    
    // Start Updating Book
    printf("{{{{{{{{{{{{{  NEW BOOK UPDATE }}}}}}}}}}}}}\n");
    int bz1[10] = {1,2,3,4,5,5,4,3,2,1};
    int ap1 = 1029, bp1 = 1028;
    double mp1 = 1028.5;
    ggoh.getTimeInfo(1., 0.2);
    ggoh.bookUpdateSim(bz1, ap1, bp1, mp1);
    ggoh.cancelAndSendNewOrders();
    ggoh.showBasicInfo();
    ggoh.showOrderBookInfo();
    
    // New book Update
    printf("{{{{{{{{{{{{{  NEW BOOK UPDATE }}}}}}}}}}}}}\n");
    int bz2[10] = {1, 3, 5, 7, 9, 11, 11, 9, 7, 5};
    int ap2 = 1030, bp2 = 1029;
    double mp2 = 1029.5;
    ggoh.getTimeInfo(1.3, 0.2);
    ggoh.bookUpdateSim(bz2, ap2, bp2, mp2);
    ggoh.cancelAndSendNewOrders();
    ggoh.showBasicInfo();
    ggoh.showOrderBookInfo();
    
    // New book Update
    printf("{{{{{{{{{{{{{  NEW BOOK UPDATE }}}}}}}}}}}}}\n");
    int bz3[10] = {3, 23, 4, 2, 3, 1, 9, 9, 7, 5};
    int ap3 = 1032, bp3 = 1031;
    double mp3 = 1031.5;
    ggoh.getTimeInfo(1.3, 0.2);
    ggoh.bookUpdateSim(bz3, ap3, bp3, mp3);
    ggoh.cancelAndSendNewOrders();
    ggoh.showBasicInfo();
    ggoh.showOrderBookInfo();
    
    // trade
    printf("{{{{{{{{{{{{{  TRADE COMES }}}}}}}}}}}}}\n");
    int tv = 32;
    ggoh.getTimeInfo(1.6, 0.2);
    ggoh.processTrade(true, 1028, tv);
    ggoh.cancelAndSendNewOrders();
    ggoh.showBasicInfo();
    ggoh.showOrderBookInfo();
    
    // Test finished
    printf("\n <<< TEST FINISHED !!! >>>\n");
}

int main(){
    def_dvec(float) dev_out(1, 0);
    test<<<1, 1>>>();
    return 0;
}