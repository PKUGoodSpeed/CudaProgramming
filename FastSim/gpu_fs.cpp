#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <cassert>
#include <ctime>
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
            prof[global_i] -= last_prc[global_i];
            pos[global_i] += 1;
            i += late[i];
        }
        else if(alpha[i]*mid[i%N_batch]<-gap[i%N_batch] && pos[global_i]>-1){
            last_prc[global_i] = mid[i%N_batch] - gap[i%N_batch];
            prof[global_i] += last_prc[global_i];
            pos[global_i] -= 1;
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
    for(int i=0;i<N_stgy;++i){
        for(int j=0;j<N_batch;++j) cout<<dev_C[i*N_batch + j]<<"\t";
        cout<<endl;
    }
    
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
    for(int i=0;i<N_samp;i+=N_batch) this->operator()(i*N_batch, min(N_batch, N_samp - i*N_batch));
    this->finalizeSim();
    return;
}

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
    vector<vector<double>> weights = {
        {1., 1., 0., 0.},
        {0., 0., -1., 0.},
        {0., 0., 0., -1.}
    };
    FastSim<gpu, double> test(signals, prices);
    vector<int> late = {1, 1, 1, 1, 1, 1, 1, 1};
    auto res = test.getPerfectOps(late);
    cout<<"The optimal operation list is: \n";
    for(auto k:res) cout<<k<<' ';
    cout<<endl;
    test.fastSimulation(weights, late, 4);
    return 0;
}
