#include <iostream>
#include "../ginkgo/GOrder.h"
#include "../include/lglist.h"
#include <thrust/device_vector.h>
#define def_dvec(t) thrust::device_vector<t>

using namespace std;
const int MAX_LENGTH = 20;

typedef gpu_ginkgo::Order gorder;

__device__ void printList(gpu_linearized_stl::list<gpu_ginkgo::Order, MAX_LENGTH> &ol){
    if(ol.empty()){
        printf("\n===== This is an empty order list =====\n\n");
        return;
    }
    for(auto p=ol.begin(); p!=ol.end(); ol.increment(p)) ol.at(p).showOrderInfo();
    printf("\n");
}

__global__ void test(){
    gpu_linearized_stl::list<gpu_ginkgo::Order, MAX_LENGTH> ol;
    printList(ol);
    //
    printf("Adding New Order\n");
    ol.push_back(gorder(1024, 15, 100, 0));
    printf("ol.push_back(gorder(1024, 15, 100, 0));\n");
    printList(ol);
    //
    printf("Adding New Order\n");
    ol.push_back(gorder(1024, 15, 101, 15));
    printf("ol.push_back(gorder(1024, 15, 101, 15));\n");
    printList(ol);
    //
    printf("Adding New Order\n");
    ol.push_back(gorder(1024, 15, 102, 30));
    printf("ol.push_back(gorder(1024, 15, 102, 30));\n");
    printList(ol);
    //
    printf("Adding New Order\n");
    ol.push_back(gorder(1024, 15, 103, 45));
    printf("ol.push_back(gorder(1024, 15, 103, 45));\n");
    printList(ol);
    //
    int p = ol.begin();
    printf("Acking The first Order\n");
    ol.at(p).getAcked(25);
    printf("int p = ol.begin();ol.at(p).getAcked(25);");
    printList(ol);
    //
    ol.at(p).qUpdateAgainstBU(50);
    ol.increment(p);
    ol.at(p).getAcked(50);
    printf("ol.at(p).qUpdateAgainstBU(50);ol.increment(p);ol.at(p).getAcked(50);");
    printList(ol);
    //
    ol.at(ol.begin()).qUpdateAgainstBU(75);
    ol.at(p).qUpdateAgainstBU(75);
    ol.increment(p);
    ol.at(p).getAcked(75);
    printf("ol.at(p).qUpdateAgainstBU(75);ol.increment(p);ol.at(p).getAcked(75);");
    printList(ol);
    //
    int j = ol.begin();
    for(int i=0;i<3;i++,ol.increment(j)) ol.at(j).qUpdateAgainstBU(100);
    ol.at(j).getAcked(100);
    printf("for(int i=0, j= ol.begin();i<3;i++,ol.increment(j)) ol.at(j).qUpdateAgainstBU(100);ol.at(j).getAcked(100);");
    printList(ol);
    //
    printf("Book Volume increases to 150\n");
    j = ol.begin();
    for(int i=0;i<4;i++,ol.increment(j)) ol.at(j).qUpdateAgainstBU(150);
    printf("for(int i=0;i<4;i++,ol.increment(j)) ol.at(j).qUpdateAgainstBU(150);");
    printList(ol);
    //
    printf("Book Volume decreases to 45\n");
    j = ol.begin();
    for(int i=0;i<4;i++,ol.increment(j)) ol.at(j).qUpdateAgainstBU(45);
    printf("for(int i=0;i<4;i++,ol.increment(j)) ol.at(j).qUpdateAgainstBU(45);");
    printList(ol);
    //
    printf("A trade of 60 with price == 1024 is coming against us!!!\n");
    int filled_qty = 0, filled_pnl = 0, dq = 0, dqs = 0, tz = 60, prc = 1024;
    int bz = 0;
    
    for(j=ol.begin(); j!=ol.end();){
        if(ol.at(j).price != prc){
            ol.increment(j);
            continue;
        }
        ol.at(j).qUpdateAgainstTrade(dq, dqs);
        if(ol.at(j).filledAgainstTrade(tz, filled_qty, filled_pnl, dq, dqs, 1024, bz)){
            j = ol.erase(j);
        }
        else ol.increment(j);
    }
    printList(ol);
}

int main(){
    def_dvec(float) dev_out(1, 0);
    test<<<1, 1>>>();
    return 0;
}