#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include "fastsim.hpp"

using namespace std;

int main(int argc, char *argv[]){
    assert(argc > 1);
    ifstream fin;
    ofstream fout;
    fin.open(argv[1]);
    fout.open("perfect_action.txt");
    string info;
    getline(fin, info);
    int N_samp = 1E6, N_feat = 11;
    vector<vector<double>> prices(2, vector<double>(N_samp)), signals(N_feat, vector<double>(N_samp,0));
    vector<int> late(N_samp, 5);
    for(int i=0;i<N_samp;++i){
        getline(fin, info);
        auto j = info.find(',') + 1;
        double mid = stod(info.substr(j));
        j = info.find(',',j) + 1;
        for(int k=0;k<11;++k){
            signals[k][i] = stod(info.substr(j));
            j = info.find(',',j) + 1;
        }
        double gap = stod(info.substr(j));
        prices[0][i] = mid - gap/2.;
        prices[1][i] = mid + gap/2.;
    }
    FastSim<cpu, double> test(signals, prices);
    clock_t t_start = clock(), t_end;
    auto res = test.getPerfectOps(late);
    t_end = clock();
    fout<< setprecision(12);
    for(auto k:res) fout<<k<<' ';
    fout<<endl;
    for(int i=0;i<N_samp;++i) fout<<prices[0][i]<<' ';
    fout<<endl;
    for(int i=0;i<N_samp;++i) fout<<prices[1][i]<<' ';
    fout<<endl;
    cout<<"Time usage is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl;
    return 0;
}
