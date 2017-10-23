#ifndef fastsim_h
#define fastsim_h

#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;
struct cpu{};
struct gpu{};
template <typename DEVICE_TYPE, typename DATA_TYPE>
class FastSim{
    int N_samp, N_feat, N_stgy;
    vector<DATA_TYPE> mid, gap, stgy;
    vector<vector<DATA_TYPE>> signals;
    // The following vectores are used to mark the status of each simulation trajectory
    vector<DATA_TYPE> prof, last_prc;
    vector<int> pos, rest_lag, trd_cnt;
    // The following vector latencies is used to store the latency information
    vector<int> latencies;
    const DATA_TYPE worst = -1E7;
public:
    FastSim(const vector<vector<DATA_TYPE>> &sigs, const vector<vector<DATA_TYPE>> &prices):N_stgy(0), stgy(vector<DATA_TYPE>()), signals(sigs){
        assert(!sigs.empty() && !sigs[0].empty() && !prices.empty() && (int)prices.size() == 2);
        assert(sigs[0].size() == prices[0].size());
        N_samp = (int)sigs[0].size();
        N_feat = (int)sigs.size();
        mid.resize(N_samp);
        gap.resize(N_samp);
        transform(prices[0].begin(), prices[0].end(), prices[1].begin(), mid.begin(), [](DATA_TYPE x, DATA_TYPE y){return abs(y+x)/2;});
        transform(prices[0].begin(), prices[0].end(), prices[1].begin(), gap.begin(), [](DATA_TYPE x, DATA_TYPE y){return abs(y-x)/2;});
        latencies = vector<int>(N_samp, 0);
    }
    /* loading the strategy weights into this objects */
    void loadWeights(const vector<vector<DATA_TYPE>> &weights){
        assert(!weights.empty() && !weights[0].empty());
        assert((int)weights[0].size() == N_feat);
        N_stgy = (int)weights.size();
        stgy.resize(N_stgy * N_feat);
        for(int i=0;i<N_stgy;++i) copy(weights[i].begin(), weights[i].end(),stgy.begin() + i*N_feat);
        rest_lag = pos = trd_cnt = vector<int>(N_stgy, 0);
        last_prc = prof = vector<DATA_TYPE>(N_stgy, 0.);
    }
    
    /* loading the latency information */
    void loadLatencies(const vector<int> &late){
        assert((int)late.size() == N_samp);
        copy(late.begin(), late.end(), latencies.begin());
    }
    
    /* Do fast simulation for a batch of data */
    void operator ()(const int &start_pos, const int &N_batch);
    void finalizeSim(){
        transform(pos.begin(), pos.end(), last_prc.begin(), last_prc.begin(), [](int x, DATA_TYPE y){return (DATA_TYPE)x*y;});
        transform(prof.begin(), prof.end(), last_prc.begin(), prof.begin(), plus<DATA_TYPE>());
        transform(trd_cnt.begin(), trd_cnt.end(), trd_cnt.begin(), [](int x){if(x>1) return x; else return 0;});
        cout<<"Showing the PNL Results:"<<endl;
        for(auto k:prof) cout<<k<<"\t";
        cout<<endl;
        cout<<"Showing the Trade Count Results:"<<endl;
        for(auto k:trd_cnt) cout<<k<<"\t";
        cout<<endl;
        return;
    }
    void fastSimulation(const vector<vector<DATA_TYPE>> &weights, const vector<int> &late, const int &N_batch);
    
    /* The following function is used to check whether the matrix multiplication is done correctly */
    vector<DATA_TYPE> testMatMul(const vector<DATA_TYPE> &A, const vector<DATA_TYPE> &B,int rA, int cA, int cB){
        assert((int)A.size() == rA * cA);
        assert((int)B.size() == cA * cB);
        vector<DATA_TYPE> C(rA*cB, 0);
        for(int i=0;i<rA;++i) for(int j=0;j<cB;++j) for(int k=0;k<cA;++k) C[i*cB + j] += A[i*cA + k]*B[k*cB + j];
        return C;
    }
    vector<DATA_TYPE> testFastSim(){
        vector<DATA_TYPE> ans(N_stgy, 0.);
        for(int k=0;k<N_stgy;++k){
            vector<DATA_TYPE> weights(stgy.begin() + k*N_feat, stgy.begin() + (k+1)*N_feat);
            for(int i=0, p = 0;i<N_samp;++i){
                double f =0.;
                for(int j=0;j<N_feat;++j) f+=weights[j]*signals[j][i];
                if(f > gap[i]/mid[i] && p<1){
                    ans[k] -= (1-p)*(mid[i] + gap[i]);
                    p = 1;
                    i += latencies[i];
                }
                else if(f<-gap[i]/mid[i] && p>-1){
                    ans[k] += (p+1)*(mid[i] - gap[i]);
                    p = -1;
                    i += latencies[i];
                }
            }
        }
        return ans;
    }
};


#endif
