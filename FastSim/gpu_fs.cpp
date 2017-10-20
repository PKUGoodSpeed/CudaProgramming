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
typedef thrust::device_vector<double> dvd;
typedef thrust::device_vector<int> dvi;

const int BLOCK_SIZE = 512;

__global__ void simKernel(int N_stgy, int N_batch, double *alpha, double *mid, double *gap, int *late, int *pos, int *rest_lag, double *prof, double *last_prc){
    int global_i = blockIdx.x*blockDim.x + threadIdx.x;
    if( global_i >= N_stgy) return;
    int start = global_i*N_batch + rest_lag[global_i], end = global_i*N_batch + N_batch, i;
    for(i = start; i<end; ++i) if(alpha[i]*mid[i%N_batch]>gap[i%N_batch] || alpha[i]*mid[i%N_batch]<-gap[i%N_batch]){
        if(alpha[i]*mid[i%N_batch]>gap[i%N_batch] && pos[global_i]<1){
            last_prc[global_i] = mid[i%N_batch] + gap[i%N_batch];
            prof[global_i] -= (1-pos[global_i])*last_prc[global_i];
            pos[global_i] = 1;
            i += late[i];
        }
        else if(alpha[i]*mid[i%N_batch]<-gap[i%N_batch] && pos[global_i]>-1){
            last_prc[global_i] = mid[i%N_batch] - gap[i%N_batch];
            prof[global_i] += (pos[global_i]+1)*last_prc[global_i];
            pos[global_i] = -1;
            i += late[i];
        }
    }
    rest_lag[global_i] = i-end;
}

template<>
void FastSim<gpu, double>::operator ()(const int &start_pos, const int &N_batch){
    assert(start_pos + N_batch <= N_samp);
    dvd dev_A = stgy, dev_B(N_feat * N_batch), dev_C(N_stgy * N_batch);
    
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
    
    cout<<"Finish Matrix Multiplication"<<endl<<endl;
    
    // Initialization of GPU memories
    dvd dev_mid(N_batch), dev_gap(N_batch), dev_prof = prof, dev_prc = last_prc;
    dvi dev_pos = pos, dev_res = rest_lag, dev_late(N_batch);
    thrust::copy(mid.begin()+start_pos, mid.begin()+start_pos+N_batch, dev_mid.begin());
    thrust::copy(gap.begin()+start_pos, gap.begin()+start_pos+N_batch, dev_gap.begin());
    thrust::copy(latencies.begin()+start_pos, latencies.begin()+start_pos+N_batch, dev_late.begin());
    
    // Doing parallelized fast simulation
    simKernel<<<(N_stgy + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(N_stgy, N_batch, to_ptr(dev_C), to_ptr(dev_mid), to_ptr(dev_gap),
                                                                    to_ptr(dev_late), to_ptr(dev_pos), to_ptr(dev_res), to_ptr(dev_prof), to_ptr(dev_prc));
    for(int i=0;i<N_stgy;++i){
        cout<<dev_prof[i]<<' '<<dev_pos[i]<<' '<<dev_prc[i]<<' '<<dev_res[i]<<endl;
    }
    
    // Copy status to CPU
    thrust::copy(dev_pos.begin(), dev_pos.end(), pos.begin());
    thrust::copy(dev_prof.begin(), dev_prof.end(), prof.begin());
    thrust::copy(dev_res.begin(), dev_res.end(), rest_lag.begin());
    thrust::copy(dev_prc.begin(), dev_prc.end(), last_prc.begin());
    return;
}

template <>
void FastSim<gpu, double>::fastSimulation(const vector<vector<double>> &weights, const vector<int> &late, const int &N_batch){
    this->loadWeights(weights);
    this->loadLatencies(late);
    for(int i=0;i<N_samp;i+=N_batch) this->operator()(i, min(N_batch, N_samp - i));
    this->finalizeSim();
    return;
}

int main(int argc, char *argv[]){
    assert(argc > 1);
    ifstream fin;
    fin.open(argv[1]);
    string info;
    getline(fin, info);
    int N_samp = 1E6, N_feat = 11, N_stgy = 400;
    vector<vector<double>> prices(2, vector<double>(N_samp)), signals(N_feat, vector<double>(N_samp,0));
    clock_t t_start = clock();
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
    FastSim<gpu, double> test(signals, prices);
    clock_t t_end = clock();
    cout<<"Time usage for reading the data is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    t_start = clock();
    auto res = test.getPerfectOps(late);
    t_end = clock();
    cout<<"Time usage for computing perfect action list is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    cout<<"Testing GPU fast sim performance:\n";
    cout<<"Randomly generating weights from 0~1"<<endl;
    t_start = clock();
    vector<vector<double>> weights(N_stgy, vector<double>(N_feat));
    for(int i=0;i<N_stgy;++i) generate(weights[i].begin(), weights[i].end(), [](){return (double)rand()/RAND_MAX;});
    t_end = clock();
    cout<<"Time usage for generating the weights is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    t_start = clock();
    test.fastSimulation(weights, late, N_samp);
    t_end = clock();
    cout<<"Time usage for gpu fast sim is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    return 0;
}
