#include "../ginkgo/loaddata.hpp"
#include "../ginkgo/GOrder.h"
#include "../ginkgo/GOrderList.h"
#include "../ginkgo/GOrderHandler.h"
#include "../include/lglist.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define gpu_seq(x) thrust::sequence((x).begin(), (x).end())
#define def_dvec(t) thrust::device_vector<t>

using namespace std;
const int level_lim = 90;
const int order_lim = 100;

__global__ void simKernel(int N_tstamp, int base_p, int *booksize, int *ask, int *bid, 
int *tprice, int *tsize, int *tside, float *t_stamp, float *ltcy, int *ans){
    int max_position = 15;
    int level_order_lim = 5;
    gpu_ginkgo::OrderHandler<order_lim, level_lim> ohandler(base_p, level_order_lim);
    ohandler.loadStrategy(max_position, 0., 0.);
    for(int t=0;t<N_tstamp;++t){
        ohandler.getTimeInfo(t_stamp[t], ltcy[t]);
        if(!tside[t]){
            ohandler.bookUpdateSim(booksize+t*level_lim, ask[t], bid[t], 0.5*(ask[t] + bid[t]));
            ohandler.cancelAndSendNewOrders();
        }
        else{
            bool sell = (tside[t] == -1);
            ohandler.processTrade(sell, tprice[t], tsize[t]);
            ohandler.cancelAndSendNewOrders();
        }
        if(t%100 == 0){
            ohandler.showBasicInfo();
        }
    }
    ans[0] = ohandler.total_pnl;
    ans[1] = ohandler.pos;
    ans[2] = ohandler.total_qty;
}

int main(int argc, char* argv[]){
    assert(argc > 1);
    LoadData ld(argv[1], 0.1);
    int Level_lim = ld.preProcess();
    auto bzs = ld.getBookSize();
    auto ask = ld.getAsk();
    auto bid = ld.getBid();
    auto tsz = ld.getTradeSize();
    auto tsd = ld.getTradeSide();
    auto tp = ld.getTradePrice();
    auto tstamps = ld.getTimeStamp();
    auto base_p = ld.getBasePrice();
    int Ns = (int)ask.size();
    cout<<endl;
    cout<<ask[0]<<' '<<bid[0]<<endl;
    cout<<tsz[0]<<' '<<tsd[0]<<' '<<tp[0]<<endl;
    cout<<tstamps[0]<<endl;
    cout<<Level_lim <<' '<<base_p<<endl;
    cout<<"====================== Start simulation ======================"<<endl<<endl;
    
    def_dvec(int) d_bz(Ns * level_lim, 0), d_ap(Ns), d_bp(Ns), d_tsz(Ns), d_tp(Ns), d_tsd(Ns);
    def_dvec(float) d_t(Ns), d_ltcy(Ns, 0.);
    gpu_copy(ask, d_ap);
    gpu_copy(bid, d_bp);
    gpu_copy(tsz, d_tsz);
    gpu_copy(tsd, d_tsd);
    gpu_copy(tp, d_tp);
    gpu_copy(tstamps, d_t);
    for(int i=0;i<Ns;++i){
        gpu_copy_to(bzs[i], d_bz, i*level_lim);
    }
    def_dvec(int) ans(3, 0);
    cudaEvent_t start, stop;
    float cuda_time;
    cudaEventCreate(&start);   // creating the event 1
    cudaEventCreate(&stop);    // creating the event 2
    
    cudaEventRecord(start, 0);
    
    //Running the kernel
    simKernel<<<1, 1>>>(Ns, base_p, to_ptr(d_bz), to_ptr(d_ap), to_ptr(d_bp),
    to_ptr(d_tp), to_ptr(d_tsz), to_ptr(d_tsd), to_ptr(d_t), to_ptr(d_ltcy), to_ptr(ans));
    
    cudaEventRecord(stop, 0);                  // Stop time measuring
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop); // Saving the time measured
    cout<<"Time Usage for sim is: "<<cuda_time/1000<<"s"<<endl;
    cout<<"Total pnl = "<<ans[0]<<endl;
    cout<<"Current Position = "<<ans[1]<<endl;
    cout<<"Total trades = "<<ans[2]<<endl;
    return 0;
}