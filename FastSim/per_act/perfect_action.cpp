#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include "perfect_action.hpp"
using namespace std;

PerfectAction::PerfectAction(const vector<double> &mid_price, const vector<double> &half_spread): mid(mid_price), gap(half_spread){
    assert(!mid_price.empty());
    assert(mid_price.size() == half_spread.size());
    N_samp = (int)mid_price.size();
}

vector<int> PerfectAction::getPerfectActions(int lim, double fee, const vector<int> &lat){
    /* Using this method, we might be able to avoid using fast sim to get the optimal strategy */
    /* The perfect strategy may be not unique, this algo just finds one of them. */
    assert((int)lat.size() == N_samp);
    int n_delay = 0, n_state = 0, n_lvl = 2*lim + 1;
    for(auto l:lat) n_delay = max(n_delay, l+2);
    n_state = n_lvl * n_delay;
    vector<vector<double>> dp(N_samp, vector<double>(n_state, worst));
    vector<vector<int>> pre(N_samp, vector<int>(n_state, 0)); // back track the previous state
    /*
     state:
     for j in [k*n_delay, (k+1)*n_delay) denote the states for position = k - lim;
     j%n_delay is the latency time we need to wait until the next order
     */
    dp[0][lim*n_delay] = 0.;
    dp[0][(lim-1)*n_delay + lat[0]] = mid[0]- gap[0] - fee;
    dp[0][(lim+1)*n_delay + lat[0]] = - mid[0] - gap[0] - fee;
    for(int i=1;i<N_samp;++i){
        // If we do nothing for t_i, then the position will not change
        for(int k=0;k<n_lvl;++k){
            // We first check for the 0 latency case
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
        // Sending a buying order (We assume that the order size is always 1), then the corrent position should not be -lim
        int lag = lat[i];
        double price = mid[i] + gap[i] + fee;
        for(int k=1;k<n_lvl;++k){
            if(dp[i-1][(k-1)*n_delay] - price > dp[i][k*n_delay + lag]){
                dp[i][k*n_delay + lag] = dp[i-1][(k-1)*n_delay] - price;
                pre[i][k*n_delay + lag] = (k-1)*n_delay;
            }
        }
        // Sending a selling order (We assume that the order size is always 1), then the corrent position should not be +lim
        price = mid[i] - gap[i] - fee;
        for(int k=0;k<n_lvl-1;++k){
            if(dp[i-1][(k+1)*n_delay] + price >= dp[i][k*n_delay + lag]){
                dp[i][k*n_delay + lag] = dp[i-1][(k+1)*n_delay] + price;
                pre[i][k*n_delay + lag] = (k+1)*n_delay;
            }
        }
    }
    int cur = lim*n_delay;
    for(int j=lim*n_delay+1;j<(lim+1)*n_delay;++j) if(dp[N_samp-1][j] > dp[N_samp-1][cur]) cur = j;
    vector<int> positions{cur}, actions(N_samp);
    for(int i=N_samp-1;i>0;--i){
        cur = pre[i][cur];
        positions.push_back(cur);
    }
    reverse(positions.begin(), positions.end());
    for(int i=0;i<N_samp;++i){
        if(!i) actions[i] = positions[i]/n_delay - lim;
        else actions[i] = positions[i]/n_delay - positions[i-1]/n_delay;
    }
    return actions;
}
