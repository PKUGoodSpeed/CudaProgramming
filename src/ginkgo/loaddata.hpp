#ifndef loaddata_hpp
#define loaddata_hpp
#include <cassert>
#include <bits/stdc++.h>
using namespace std;

class LoadData{
    /*
    Here we define a class which can convert 
    csv file extracted from gsampler into data
    matrix that we need.
    The columns of the csv file are:
    ap0, az0, ap1, az1, ap2, az2, ap3, az3, ap4, az4,
    bp0, bz0, bp1, bz1, bp2, bz2, bp3, bz3, bp4, bz4,
    trade_price, trade_side, trade_volume, time
    */
    const float delta = 0.02;
    int N_feat, N_tstamp;
    vector<vector<float>> raw;
    vector<vector<int>> book_sizes;
    vector<float> tstamps;
    vector<int> tside, tprice, tsize, ask, bid;
    float max_p, min_p, tick_size;
public:
    LoadData(char* filename, const float&ts = 0.1):
    N_feat(24),
    max_p(0.),
    min_p(1.E9),
    tick_size(ts) {
        ifstream fin;
        fin.open(filename);
        string info;
        getline(fin, info);
        while(getline(fin, info)){
            auto j = info.find(',') + 1;
            vector<float> tmp;
            for(int i=0;i<23;++i){
                assert(j != string::npos);
                tmp.push_back(stof(info.substr(j)));
                j = info.find(',', j) + 1;
            }
            max_p = max(max_p, tmp[8] + 10*tick_size);
            if(tmp[18] > delta) min_p = min(min_p, tmp[18] - 10*tick_size);
            raw.push_back(tmp);
            assert(j != string::npos);
            tstamps.push_back(stof(info.substr(j))/1.E6);
        }
        fin.close();
        N_tstamp = (int)tstamps.size();
    }
    
    /* Getting level from the price */
    int getLevel(const float&prc){
        return floor((prc + delta - min_p)/tick_size);
    }
    
    /* Get Base Price */
    int getBasePrice(){
        return floor((min_p + delta)/tick_size);
    }
    
    /* Convert booksize data into integer arrays*/
    int preProcess(){
        int N_level = this->getLevel(max_p) + 1;
        book_sizes = vector<vector<int>>(N_tstamp, vector<int>(N_level, 0));
        ask.resize(N_tstamp);
        bid.resize(N_tstamp);
        tside.resize(N_tstamp);
        tsize.resize(N_tstamp);
        tprice.resize(N_tstamp);
        for(int i=0;i<N_tstamp;++i){
            for(int j=0;j<20;j+=2){
                book_sizes[i][this->getLevel(raw[i][j])] = floor(raw[i][j+1] + delta);
            }
            tsize[i] = floor(raw[i][22] + delta);
            tprice[i] = floor((raw[i][20]+delta)/tick_size);
            if(raw[i][21] > 0.5) tside[i] = 1;
            else if(raw[i][21] < -0.5) tside[i] = -1;
            else tside[i] = 0;
            ask[i] = floor((raw[i][0]+delta)/tick_size);
            bid[i] = floor((raw[i][10]+delta)/tick_size);
        }
        return N_level;
    }
    
    /* Get Book Size */
    vector<vector<int>> getBookSize(){
        return book_sizes;
    }
    
    /* Get Ask Price*/
    vector<int> getAsk(){
        return ask;
    }
    
    /* Get Bid Price*/
    vector<int> getBid(){
        return bid;
    }
    
    /* Get Trade Side*/
    vector<int> getTradeSide(){
        return tside;
    }
    
    /* Get Trade Price */
    vector<int> getTradePrice(){
        return tprice;
    }
    
    /* Get Trade Size */
    vector<int> getTradeSize(){
        return tsize;
    }
    
    /* Get Time stamps */
    vector<float> getTimeStamp(){
        return tstamps;
    }
};

#endif