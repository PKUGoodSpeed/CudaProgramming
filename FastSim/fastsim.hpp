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
    vector<int> pos, rest_lag;
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
    vector<int> getPerfectOps(const vector<int>& late){
        /* Using this method, we might be able to avoid using fast sim to get the optimal strategy */
        /* The perfect strategy may be not unique, this algo just finds one of them. */
        assert((int)late.size() == N_samp);
        int n_delay = 0, n_state = 0;
        for(auto l:late) n_delay = max(n_delay, l+1);
        n_state = 3 * n_delay;
        vector<vector<DATA_TYPE>> dp(N_samp, vector<DATA_TYPE>(n_state, worst));
        vector<vector<int>> pre(N_samp, vector<int>(n_state, 0));
        /*
         state:
         [0*n_delay, 1*n_delay) : position = -1
         [1*n_delay, 2*n_delay) : position = 0
         [2*n_delay, 3*n_delay) : position = 1
         */
        dp[0][n_delay] = 0;
        dp[0][late[0]] = mid[0] - gap[0];
        dp[0][2*n_delay + late[0]] = - mid[0] - gap[0];
        for(int i=1;i<N_samp;++i){
            // updating each slot
            
            // Doing nothing for all positions
            for(int k=0;k<=2;++k){
                if(dp[i-1][k*n_delay] > dp[i-1][k*n_delay + 1]){
                    dp[i][k*n_delay] = dp[i-1][k*n_delay];
                    pre[i][k*n_delay] = k*n_delay;
                }
                else{
                    dp[i][k*n_delay] = dp[i-1][k*n_delay + 1];
                    pre[i][k*n_delay] = k*n_delay + 1;
                }
                for(int j=1;j<n_delay-1;++j){
                    dp[i][k*n_delay + j] = dp[i-1][k*n_delay + j + 1];
                    pre[i][k*n_delay + j] = k*n_delay + j + 1;
                }
            }
            // Sending a buying order (We assume that the order size is always 1), the current position must be 0 or +1
            int lag = late[i];
            DATA_TYPE price = mid[i] + gap[i];
            for(int k=1;k<=2;++k){
                if(dp[i-1][(k-1)*n_delay] - price > dp[i][k*n_delay + lag]){
                    dp[i][k*n_delay + lag] = dp[i-1][(k-1)*n_delay] - price;
                    pre[i][k*n_delay + lag] = (k-1)*n_delay;
                }
            }
            // Sending a selling order (We assume that the order size is always 1), the current position must by -1 or 0
            price = mid[i] - gap[i];
            for(int k=0;k<=1;++k){
                if(dp[i-1][(k+1)*n_delay] + price > dp[i][k*n_delay + lag]){
                    dp[i][k*n_delay + lag] = dp[i-1][(k+1)*n_delay] + price;
                    pre[i][k*n_delay + lag] = (k+1)*n_delay;
                }
            }
        }
        int st = n_delay;
        for(int j=n_delay+1;j<2*n_delay;++j) if(dp[N_samp-1][j] > dp[N_samp-1][st]) st = j;
        vector<int> res{st}, ans(N_samp);
        for(int i=N_samp-1;i>0;--i){
            st = pre[i][st];
            res.push_back(st);
        }
        reverse(res.begin(), res.end());
        for(int i=0;i<N_samp;++i){
            if(!i) ans[i] = res[i]/n_delay - 1;
            else ans[i] = res[i]/n_delay - res[i-1]/n_delay;
        }
        return ans;
    }
    
    /* loading the strategy weights into this objects */
    void loadWeights(const vector<vector<DATA_TYPE>> &weights){
        assert(!weights.empty() && !weights[0].empty());
        assert((int)weights[0].size() == N_feat);
        N_stgy = (int)weights.size();
        stgy.resize(N_stgy * N_feat);
        for(int i=0;i<N_stgy;++i) copy(weights[i].begin(), weights[i].end(),stgy.begin() + i*N_feat);
        rest_lag = pos = vector<int>(N_stgy, 0);
        last_prc = prof = vector<DATA_TYPE>(N_stgy, 0.);
    }
    
    /* loading the latency information */
    void loadLatencies(const vector<int> &late){
        assert((int)late.size() == N_samp);
        copy(late.begin(), late.end(), latencies.begin());
    }
    
    /* Do fast simulation for a batch of data */
    void operator ()(const int &start_pos, const int &N_batch);
};


#endif
