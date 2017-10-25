#include <iostream>
#include <fstream>
#include <cassert>
#include <ctime>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "./fastsim.hpp"
using namespace std;

int main(int argc, char *argv[]){
    assert(argc > 1);
    ifstream fin;
    fin.open(argv[1]);
    int N_samp, N_feat = 3, N_stgy = 1000;
    fin>>N_samp;
    cout<<"The number of timestamps is "<<N_samp<<endl;
    clock_t t_start = clock();
    vector<vector<double>> prices(2, vector<double>(N_samp)), signals(N_feat, vector<double>(N_samp));
    for(int i=0;i<N_samp;++i){
        for(int j=0;j<3;++j) fin>>signals[j][i];
        for(int j=0;j<2;++j) fin>>prices[1-j][i];
    }
    for(int i=0;i<N_samp;i+=N_samp/20+1){
        cout<<prices[0][i]<<' '<<prices[1][i]<<' '<<signals[0][i]<<' '<<signals[1][i]<<' '<<signals[2][i]<<endl;
    }
    vector<int> late(N_samp, 0);
    FastSim<cpu, double> test(signals, prices);
    clock_t t_end = clock();
    cout<<"Time usage for reading the data is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    cout<<"Testing GPU fast sim performance:\n";
    cout<<"Randomly generating weights from 0~1"<<endl;
    t_start = clock();
    vector<vector<double>> weights(N_stgy, vector<double>(N_feat));
    for(int i=0;i<N_stgy;++i){
        for(int j=0, m=i;j<N_feat; ++j){
            weights[i][j] = (0.1*(m%10) + 0.05)*3.;
            m /= 10;
        }
    }
    t_end = clock();
    for(int i=0;i<N_stgy;i+=N_stgy/25+1) cout<<weights[i][0]<<' '<<weights[i][1]<<' '<<weights[i][2]<<endl;
    cout<<"Time usage for generating the weights is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    t_start = clock();
    vector<int> cnt;
    auto prof = test.testFastSim(double(0.008), cnt);
    t_end = clock();
    cout<<"Time usage for gpu fast sim is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    for(int i=0;i<N_stgy;i+=N_stgy/9+1){
        for(int j=0;j<3;++j) cout<<weights[i][j]<<endl;
        cout<<prof[i]<<' '<<cnt[i]<<endl;
    }
    return 0;
}

