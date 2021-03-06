/*
I am implement the K7 passive strategy iin to GPU
* Each thread saw a order, and handle its own order
* Market prce are input as integer levels
* Alpha matrix is initially given
* Latency is always 2 ticks
*/
#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int BLOCK_SIZE = 1024;
const int NUM_INSTANCE = 1024;
const int NUM_LEVELS = 256;
const int LEVEL_LIM = 5;
const int MAX_ORDER_PER_LEVEL = 2;
const int MAX_POSITION = 5;
const int MAX_QTY = 1;
const int LATENCY = 2;
const float delta = 1.E-6;

/* This function is used to compute ask and bid BBOs */
__device__ int getBBO(float smp, float gap, float ts, float coeff, int pos, bool ask){
    float bbo_price = smp + coeff*pos + (ask? 1:-1)*gap;
    return int((bbo_price + (ask? -1:1)*delta)/ts) + (ask? 1:0);
}

/* The following function is used to put bad orders into cancel order list */
__device__ void cancelAskOrders(int *idx, int *info, int *start_idx, int *sz, int *qty, int k, int *c_idx,int *c_size, ){
    for(int i=0;i<min(k,LEVEL_LIM);++i){
        int start = i*MAX_ORDER_PER_LEVEL + start_idx;
        for(int j=0;j<sz[i];++j){
            
        }
    }
}

/* This function is used to update trades when the BBO level is shifted */
__device__ void shiftLevel(int *idx, int k, int *tmp){
    if(LEVEL_LIM <= abs(k)) return;
    for(int i=0;i<LEVEL_LIM;++i){
        tmp[i] = idx[(i+k+LEVEL_LIM)%LEVEL_LIM];
    }
    thrust::copy(thrust::device, tmp, tmp+LEVEL_LIM, idx);
}


__global__ void gpuPassive(int Nstamp, int Ninst, float tick_size, float *shifted_mid_price, float *pspread, float *mid_price,
float *beta, int *book_size, float *final_pnl){
    int b_sz = blockDim.x, b_id = blockIdx.x, t_id = threadIdx.x;
    int g_id = b_sz * b_id + t_id;
    int ask_info[LEVEL_LIM], bid_info[LEVEL_LIM], ask_idx[LEVEL_LIM], bid_idx[LEVEL_LIM], ask_sz[LEVEL_LIM], bid_sz[LEVEL_LIM];
    for(int i=0;i<LEVEL_LIM;++i){
        ask_info[2*i] = ask_info[2*i+1] = bid_info[2*i] = bid_info[2*i+1] = 0;
        ask_idx[i] = bid_idx[i] = i;
        ask_sz[i] = bid_sz[i] = 0;
    }
    int pos = 0;
    int pen_pos = 0;
    float pnl = 0;
    int ask_lvl = 0, bid_lvl = 0;
    for(int t=0;t<Nstamp;++t){
        float smp = shifted_mid_price[g_id * Nstamp + t], gap = 0.5*pspread[t], coeff = beta[g_id], mid = mid_price[t];
        int askBBO = (smp + gap + coeff*pen_pos)/tick_size + 1;
        int bidBBO = (smp - gap + coeff*pen_pos)/tick_size;
        int alvl = (mid + gap + 0.000001)/tick_size;
        int blvl = (mid - gap + 0.000001)/tick_size;
        for(int i=0;i<min(5, blvl - lvlask + 1);++i){
            pos += ask_qty[i];
            pnl -= ask_qty[i]*(i+lvlask)*tick_size;
            ask_qty[i] = ask_aQ[i] = ask_bQ[i] = 0;
        }
        for(int i=0;i<min(5, lvlbid - alvl + 1); ++i){
            pos -= bid_qty[i];
            pnl += bid_qty[i]*(lvlbid - i)*tick_size;
            bid_qty[i] = bid_aQ[i] = bid_bQ[i] = 0;
        }
        for(int i=0;i<5;++i) if(lvlask+i<askBBO || lvlask+i>=askBBO+5){
            pen_pos -= ask_qty[i];
            ask_qty[i] = ask_aQ[i] = ask_bQ[i] = 0;
        }
        for(int i=0;i<5;++i) if(lvlbid-i>bidBBO || lvlbid-i<=bidBBO-5){
            pen_pos += bid_qty[i];
            bid_qty[i] = bid_aQ[i] = bid_bQ[i] = 0;
        }
        for(int i = max(0, lvlask - askBBO); i< min(5, 5 + lvlask - askBBO); ++i){
            tmp_qty[i] = ask_qty[i + askBBO - lvlask];
            tmp_aQ[i] = ask_aQ[i + askBBO - lvlask];
            tmp_bQ[i] = ask_bQ[i + askBBO - lvlask];
        }
        for(int i=0;i<5;++i){
            ask_qty[i] = tmp_qty[i];
            ask_aQ[i] = tmp_aQ[i];
            ask_bQ[i] = tmp_bQ[i];
            if(!ask_qty[i] && pen_pos<10){
                ask_qty[i] = 1;
                pen_pos += 1;
            }
            tmp_qty[i] = tmp_aQ[i] = tmp_bQ[i] = 0;
        }
        for(int i=max(0, bidBBO - lvlbid); i < min(5, 5+bidBBO-lvlbid); ++i){
            tmp_qty[i] = bid_qty[i + lvlbid - bidBBO];
            tmp_aQ[i] = bid_aQ[i + lvlbid - bidBBO];
            tmp_bQ[i] = bid_bQ[i + lvlbid - bidBBO];
        }
        for(int i=0;i<5;++i){
            bid_qty[i] = tmp_qty[i];
            bid_aQ[i] = tmp_aQ[i];
            bid_bQ[i] = tmp_bQ[i];
            if(!bid_qty[i] && pen_pos>-10){
                bid_qty[i] = 1;
                pen_pos -= 1;
            }
            tmp_qty[i] = tmp_aQ[i] = tmp_bQ[i] = 0;
        }
    }
}

int main(){
    int N = 10000;
    def_dvec(int) asize(N*NUM_LEVELS, 0), bsize(N*NUM_LEVELS);
    float ts = 1.;
    def_dvec(float) aprc(N*NUM_INSTANCE, 3128.), gamma(NUM_INSTANCE, 0.), beta(NUM_INSTANCE, 0.), 
    pnl(NUM_INSTANCE, 0.);
    gpuPassive<<<NUM_INSTANCE, NUM_LEVELS>>>(N, NUM_INSTANCE, to_ptr(asize), to_ptr(bsize), 
    3000, ts, to_ptr(aprc), to_ptr(beta), to_ptr(gamma), to_ptr(pnl));
    for(auto k:pnl) cout<<k<<' ';
}
