#ifndef perfect_action_h
#define perfect_action_h

#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class PerfectAction{
    int N_samp;
    vector<double> mid, gap;
    const double worst = -1.E8;
public:
    PerfectAction(const vector<double> &mid_price, const vector<double> &spread);
    vector<int> getPerfectActions(int lim, double fee, const vector<int> &lat);
};


#endif
