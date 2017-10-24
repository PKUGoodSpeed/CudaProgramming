#include <iomanip>
#include <cmath>
#include "perfect_action.hpp"
using namespace std;

/* Methods for LoadData class */

int double_comp(double x, double y){
    if(x < y-err) return -1;
    else if(x < y+err) return 0;
    else return 1;
}

LoadData::LoadData(char *filename, int num_features): N_feat(num_features){
    features.resize(N_feat);
    fin.open(filename);
    string info;
    getline(fin, info);
    while(getline(fin, info)){
        auto j = info.find(' ') + 1;
        double h = stod(info.substr(j)), m = stod(info.substr(j+3)), s = stod(info.substr(j+6));
        tstamps.push_back(h*3600. + m*60. + s);
        for(int i=0;i<N_feat;++i){
            j = info.find(',',j + 1);
            assert(j != string::npos);
            features[i].push_back(stod(info.substr(j+1)));
        }
    }
    N_samp = (int)features[0].size();
    fin.close();
}

int LoadData::getNumSamples(){
    return N_samp;
}

vector<vector<double>> LoadData::getFeatures(){
    return features;
}

vector<vector<double>> LoadData::getNPFeatures(){
    return vector<vector<double>>(features.begin()+1, features.begin()+N_feat-1);
}

vector<double> LoadData::getMidPrices(){
    return features[0];
}

vector<double> LoadData::getPriceGaps(){
    vector<double> gap(features[N_feat-1]);
    for(auto &val:gap) val /= 2.;
    return gap;
}

vector<double> LoadData::getTimeStamps(){
    return tstamps;
}

/* Methods for ExpandFeatures class */
ExpandFeatures::ExpandFeatures(char *filename, int num_features):raw_info(filename, num_features){
    N_samp = raw_info.getNumSamples();
}

int ExpandFeatures::getNumSamples(){
    return N_samp;
}

vector<vector<double>> ExpandFeatures::getOriginalFeatures(){
    auto feats = raw_info.getNPFeatures();
    for(int i=0;i<(int)feats.size();++i){
        double sum = 0., dev = 1.;
        for(int j=0;j<(int)feats[i].size();++j){
            sum += feats[i][j];
            double mean = sum/double(j+1);
            dev += (feats[i][j] - mean)*(feats[i][j] - mean);
            feats[i][j] = (feats[i][j] - mean)/pow(dev/double(j+1), 1./2.);
        }
    }
    return feats;
}

vector<vector<double>> ExpandFeatures::getPriceFeatures(int back_track, double tick_val){
    auto mid = raw_info.getMidPrices(), gap = raw_info.getPriceGaps();
    vector<vector<double>> prc(back_track+2,vector<double>(N_samp, 0.));
    for(int j=0;j<N_samp;++j) prc[0][j] = gap[j]/tick_val;
    double sum = 0., dev = 1.;
    for(int j=0;j<N_samp;++j){
        sum += mid[j];
        double mean = sum/double(j+1);
        dev += (mid[j] - mean)*(mid[j] - mean);
        prc[1][j] = (mid[j] - mean)/pow(dev/double(j+1), 1./2.);
    }
    for(int i=2;i<back_track+2;++i) for(int j=1;j<N_samp;++j) prc[i][j] = prc[i-1][j-1];
    return prc;
}

vector<vector<double>> ExpandFeatures::getStepFeatures(double lambda){
    auto mid = raw_info.getMidPrices(), gap = raw_info.getPriceGaps();
    vector<vector<double>> turns(6, vector<double>(N_samp, 0.));
    for(int i=1;i<N_samp;++i){
        for(int j=0;j<6;++j) turns[j][i] = 0.;
        int judge = double_comp(mid[i] + gap[i], mid[i-1] + gap[i-1]);
        if(judge == -1) turns[0][i] = turns[0][i-1] + lambda;
        else if(judge == 0) turns[1][i] = turns[1][i-1] + lambda;
        else turns[2][i] = turns[2][i-1] + lambda;
        judge = double_comp(mid[i] - gap[i], mid[i-1] - gap[i-1]);
        if(judge == -1) turns[3][i] = turns[3][i-1] + lambda;
        else if(judge == 0) turns[4][i] = turns[4][i-1] + lambda;
        else turns[5][i] = turns[5][i-1] + lambda;
    }
    return turns;
}

/* Methods for PerfectAction class */

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

int main(int argc, char *argv[]){
    assert(argc > 1);
    ofstream fout;
    ExpandFeatures exfeat(argv[1], 13);
    auto orig_feats = exfeat.getOriginalFeatures();
    int N_feat = (int)orig_feats.size(), N_samp = (int)orig_feats[0].size();
    cout<<N_feat<<' '<<N_samp<<endl;
    cout<<"Generate the original features!"<<endl;
    cout<<orig_feats[0].size()<<endl;
    fout.open("feats/orig_feats.txt");
    for(int i=0;i<N_feat;++i){
        for(int j=0;j<N_samp;++j){
            fout<<orig_feats[i][j];
            if(j<N_samp-1) fout<<' ';
        }
        if(i<N_feat) fout<<endl;
    }
    fout.close();
    
    cout<<"Generate the price features!"<<endl;
    auto prc_feats = exfeat.getPriceFeatures(10, 1.);
    cout<<prc_feats[0].size()<<endl;
    fout.open("feats/prc_feats.txt");
    for(int i=0;i<12;++i){
        for(int j=0;j<N_samp;++j){
            fout<<prc_feats[i][j];
            if(j<N_samp-1) fout<<' ';
        }
        if(i<11) fout<<endl;
    }
    fout.close();
    
    cout<<"Generate the step features!"<<endl;
    auto stp_feats = exfeat.getStepFeatures(1./4.);
    cout<<stp_feats[0].size()<<endl;
    fout.open("feats/stp_feats.txt");
    for(int i=0;i<6;++i){
        for(int j=0;j<N_samp;++j){
            fout<<stp_feats[i][j];
            if(j<N_samp-1) fout<<' ';
        }
        if(i<5) fout<<endl;
    }
    fout.close();
    orig_feats.clear();
    prc_feats.clear();
    stp_feats.clear();
    
    LoadData raw_info(argv[1], 13);
    PerfectAction pact(raw_info.getMidPrices(), raw_info.getPriceGaps());
    
    // Generate outputs with different latency values
    for(int lat = 1; lat<=6; ++lat){
        cout<<"Generating one set of output with latency = "<<lat<<endl;
        string fname = "output/output_lat=" + to_string(lat) +".txt";
        fout.open(fname.c_str());
        auto ans = pact.getPerfectActions(1, 0., vector<int>(N_samp, lat));
        for(int i=0;i<N_samp;++i){
            fout<<ans[i];
            if(i<N_samp - 1) fout<<' ';
        }
        fout.close();
        cout<<ans.size()<<endl;
    }
    
    // Generate outputs with different position limit
    for(int lim = 1; lim <= 2; ++lim){
        cout<<"Generating one set of output with position limit = "<<lim<<endl;
        string fname = "output/output_lim=" + to_string(lim) +".txt";
        fout.open(fname.c_str());
        auto ans = pact.getPerfectActions(lim, 0., vector<int>(N_samp, 3));
        for(int i=0;i<N_samp;++i){
            fout<<ans[i];
            if(i<N_samp - 1) fout<<' ';
        }
        fout.close();
        cout<<ans.size()<<endl;
    }
    
    // Generate outputs with different fees
    for(double fee = 0.; fee<5.1 ; fee += 1.){
        cout<<"Generating one set of output with fee = "<<fee<<endl;
        string fname = "output/output_fee=" + to_string(fee) +".txt";
        fout.open(fname.c_str());
        auto ans = pact.getPerfectActions(1, fee, vector<int>(N_samp, 3));
        for(int i=0;i<N_samp;++i){
            fout<<ans[i];
            if(i<N_samp - 1) fout<<' ';
        }
        fout.close();
        cout<<ans.size()<<endl;
    }
    
    // Print out price information
    auto p_mid = raw_info.getMidPrices(), p_gap = raw_info.getPriceGaps();
    cout<<"Print out price information"<<endl;
    fout.open("feats/prices.txt");
    for(int i=0;i<N_samp;++i){
        fout<<p_mid[i]+p_gap[i];
        if(i<N_samp-1) fout<<' ';
        else fout<<endl;
    }
    for(int i=0;i<N_samp;++i){
        fout<<p_mid[i]-p_gap[i];
        if(i<N_samp-1) fout<<' ';
    }
    fout.close();
    cout<<"DONE!"<<endl;
    return 0;
}
