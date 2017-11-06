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
const int MAX_POSITION = 5;
const int MAX_QTY = 1;
const int LATENCY = 2;

__device__ float getAskBbo(float alpha_price, float beta, float gamma, int pos, int Q){
    return alpha_price + beta*pos + gamma*Q;
}

__device__ float getBidBbo(float alpha_price, float beta, float gamma, int pos, int Q){
    return alpha_price + beta*pos - gamma*Q;
}

__global__ void gpuPassive(int N, int N_ins, int *ask_size, int *bid_size, int lvl0, float ts, float *alp_prc, float *beta, float *gamma, float *pnl){
    int b_sz = blockDim.x, b_id = blockIdx.x, t_id = threadIdx.x;
    __shared__ int info[3];
    __shared__ float bbo[2];
    if(t_id < 3){
        info[t_id] = 0;
    }
    float prc = (lvl0 + t_id) * ts;
    int latency = 0, qty = 0, filled_qty = 0, Q = 0, bsz = 0, asz = 0;
    bool sell = true, cancel = false, sending = true;
    __syncthreads();
    for(int i=0; i<N; ++i){
        /* compute bbos */
        if(t_id < 2){
            float tmp = (t_id? -1:1)*gamma[b_id];
            bbo[t_id] = getAskBbo(alp_prc[b_id*N + i], beta[b_id], tmp, info[0], info[1]);
            info[1] = 0;
        }
        __syncthreads();
        
        /* Checking filled status */
        // If an order latency just arrives, initialize it
        asz = ask_size[NUM_LEVELS*i + t_id]; 
        bsz = bid_size[NUM_LEVELS*i + t_id];
        if(sending && !latency){
            sending = false;
            Q = (sell? asz:bsz);
            info[0] += (sell? -1:1)*qty;
        }
        // If the order is ready to fill, fill it
        if(!sending && qty){
            Q -= (sell? bsz:asz);
            filled_qty = min(qty, max(0, -Q));
            qty -= filled_qty;
            info[2] += (sell? 1:-1)*filled_qty*(t_id + lvl0);
            Q = max(Q, 0);
        }
        
        /* Checing whether cancelled */
        if(cancel && !latency){
            cancel = false;
            info[0] += (sell? 1:-1)*qty;
            qty = 0;
        }
        // 1. Checking sell order for cancel.
        if(!cancel && !sending && qty && sell && (prc < bbo[0] || prc > bbo[0] + LEVEL_LIM*ts)){
            cancel = true;
            latency = LATENCY - 1;
        }
        // 2. Checing buy order for cancel
        if(!cancel && !sending && qty && !sell && (prc > bbo[1] || prc < bbo[1] - LEVEL_LIM*ts)){
            cancel = true;
            latency = LATENCY;
        }
        
        // When the order is in latency, we cannot sending an order
        if(latency) --latency;
        /* Checking whether to cancel */
        else{
            // Sending a sell order
            if(!qty && prc >= bbo[0] && prc <= bbo[0] + LEVEL_LIM && info[0]>-MAX_POSITION){
                sell = sending = true;
                qty = MAX_QTY;
                info[0] -= qty;
            }
            // Sending a buy order
            if(!qty && prc >= bbo[0] && prc <= bbo[0] + LEVEL_LIM && info[0]<MAX_POSITION){
                sell = false;
                sending = true;
                qty = MAX_QTY;
                info[0] += qty;
            }
        }
        info[1] += Q;
        __syncthreads();
    }
    pnl[b_id] = info[2]*ts;
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
