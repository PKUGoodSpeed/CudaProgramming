#ifndef perfect_action_h
#define perfect_action_h

#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

const double err = 1.E-8;
int double_comp(double x, double y);

/* Class for loading the data */
class LoadData{
    ifstream fin;
    vector<vector<double>> features;
    vector<double> tstamps;
    int N_feat;
public:
    LoadData(char *filename, int num_features);
    vector<vector<double>> getFeatures();
    vector<vector<double>> getNPFeatures();
    vector<double> getMidPrices();
    vector<double> getPriceGaps();
    vector<double> getTimeStamps();
};

/* Class for generating extended features */
class ExpandFeatures{
    LoadData raw_info;
    vector<vector<double>> features;
public:
    ExpandFeatures(char *filename, int num_features);
    void buildFeatures();
    vector<vector<double>> getExFeatures();
};

/* Class for computing the perfect actions */
class PerfectAction{
    int N_samp;
    vector<double> mid, gap;
    const double worst = -1.E8;
public:
    PerfectAction(const vector<double> &mid_price, const vector<double> &half_spread);
    vector<int> getPerfectActions(int lim, double fee, const vector<int> &lat);
};


#endif
