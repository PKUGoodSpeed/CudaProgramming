#include <iostream>
#include <vector>
#include "fastsim.hpp"

using namespace std;

int main(){
    vector<vector<double>> signals = {
        {1., 2., 3., 4., 5., 6., 7., 8.},
        {1., 2., 3., 4., 5., 6., 7., 8.},
        {1., 2., 3., 4., 5., 6., 7., 8.},
        {1., 2., 3., 4., 5., 6., 7., 8.}
    };
    vector<vector<double>> prices = {
        {3., 1., 4., 1., 5., 2., 1, 4.},
        {9., 2., 6., 3., 7., 3., 2, 7.}
    };
    FastSim<cpu, double> test(signals, prices);
    vector<int> late = {1, 1, 1, 1, 1};
    auto res = test.getPerfectOps(late);
    cout<<"The optimal operation list is: \n";
    for(auto k:res) cout<<k<<' ';
    cout<<endl;
    return 0;
}
