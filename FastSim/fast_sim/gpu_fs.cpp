#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <fstream>
#include <cassert>
#include <ctime>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include "./fastsim.hpp"

#define to_ptr(x) thrust::raw_pointer_cast(&x[0])

using namespace std;





template<>
void FastSim<gpu, double>::operator ()(const int &start_pos, const int &N_batch, double fee){
    assert(start_pos + N_batch <= N_samp);
    dvd dev_A = stgy, dev_B(N_feat * N_batch), dev_C(N_stgy * N_batch);
    clock_t t_start, t_end;
    t_start = clock();
    // First doing matrix multiplication
    for(int i=0;i<N_feat;++i) thrust::copy(signals[i].begin() + start_pos, signals[i].begin() + start_pos + N_batch, dev_B.begin() + i*N_batch);
    // Initialization of cuBlas
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if(status != CUBLAS_STATUS_SUCCESS) cerr << "CUBLAS initialization error!\n";
    
    double alpha = 1.0, beta = 0.0;
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N_batch, N_stgy, N_feat,
                         &alpha, to_ptr(dev_B), N_batch,
                         to_ptr(dev_A), N_feat,
                         &beta,  to_ptr(dev_C), N_batch);
    if (status != CUBLAS_STATUS_SUCCESS) cerr << "Kernel execution error!\n";
    // Finalization of cuBlas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) cerr << "!!!! shutdown error (A)\n";
    t_end = clock();
    cout<<"Time usage for matrix multiplication is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl;
    
    // Initialization of GPU memories
    t_start = clock();
    dvd dev_mid(N_batch), dev_gap(N_batch), dev_prof = prof, dev_prc = last_prc;
    dvi dev_pos = pos, dev_res = rest_lag, dev_late(N_batch), dev_cnt = trd_cnt;
    thrust::copy(mid.begin()+start_pos, mid.begin()+start_pos+N_batch, dev_mid.begin());
    thrust::copy(gap.begin()+start_pos, gap.begin()+start_pos+N_batch, dev_gap.begin());
    thrust::copy(latencies.begin()+start_pos, latencies.begin()+start_pos+N_batch, dev_late.begin());
    
    // Doing parallelized fast simulation
    simKernel<<<(N_stgy + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(N_stgy, N_batch, to_ptr(dev_C), to_ptr(dev_mid), to_ptr(dev_gap),
                                                                    to_ptr(dev_late), to_ptr(dev_pos), to_ptr(dev_res), to_ptr(dev_prof), to_ptr(dev_prc), to_ptr(dev_cnt), fee);
    // Copy status to CPU
    thrust::copy(dev_pos.begin(), dev_pos.end(), pos.begin());
    thrust::copy(dev_cnt.begin(), dev_cnt.end(), trd_cnt.begin());
    thrust::copy(dev_prof.begin(), dev_prof.end(), prof.begin());
    thrust::copy(dev_res.begin(), dev_res.end(), rest_lag.begin());
    thrust::copy(dev_prc.begin(), dev_prc.end(), last_prc.begin());
    t_end = clock();
    cout<<"Time usage for running simulation is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl;
    return;
}

template <>
void FastSim<gpu, double>::fastSimulation(const vector<vector<double>> &weights, const vector<int> &late, const int &N_batch, double fee){
    this->loadWeights(weights);
    this->loadLatencies(late);
    for(int i=0;i<N_samp;i+=N_batch) {
        cout<<endl<<endl<<"Batch Started"<<endl;
        this->operator()(i, min(N_batch, N_samp - i), fee);
        cout<<"Batch Finished"<<endl;
    }
    this->finalizeSim();
    return;
}

/*
int main(int argc, char *argv[]){
    assert(argc > 1);
    ifstream fin;
    fin.open(argv[1]);
    string info;
    getline(fin, info);
    int N_samp = 1E6, N_feat = 11, N_stgy = 1000;
    vector<vector<double>> prices(2, vector<double>(N_samp)), signals(N_feat, vector<double>(N_samp,0));
    clock_t t_start = clock();
    vector<int> late(N_samp, 1);
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
    FastSim<gpu, double> test(signals, prices);
    clock_t t_end = clock();
    cout<<"Time usage for reading the data is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    cout<<"Testing GPU fast sim performance:\n";
    cout<<"Randomly generating weights from 0~1"<<endl;
    t_start = clock();
    vector<vector<double>> weights(N_stgy, vector<double>(N_feat));
    for(int i=0;i<N_stgy;++i) generate(weights[i].begin(), weights[i].end(), [](){return 0.04*(double)rand()/RAND_MAX;});
    t_end = clock();
    cout<<"Time usage for generating the weights is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    t_start = clock();
    test.fastSimulation(weights, late, N_samp/10);
    t_end = clock();
    cout<<"Time usage for gpu fast sim is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    return 0;
}
 */

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
    FastSim<gpu, double> test(signals, prices);
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
    test.fastSimulation(weights, late, 200000, double(0.008));
    t_end = clock();
    cout<<"Time usage for gpu fast sim is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    return 0;
}
