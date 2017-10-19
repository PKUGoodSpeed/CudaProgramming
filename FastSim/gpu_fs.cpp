#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <cassert>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include "./fastsim.hpp"

using namespace std;
typedef thrust::device_vector<double> dvd;

const int BLOCK_SIZE = 32;
const int NUM_BLOCKS = 512;

/*
template<>
vector<float> MatrixMultiplication<cublas, float>::operator ()(const vector<float> &A, const vector<float> &B, int rA, int cA, int rB, int cB){
    assert((int)A.size() == rA * cA);
    assert((int)B.size() == rB * cB);
    assert(cA == rB);
    dvf dC(rA * cB), dA = A, dB = B;
    cublasHandle_t handle;
    clock_t t_start = clock(), t_end;

    cublasStatus_t status = cublasCreate(&handle);
    if(status != CUBLAS_STATUS_SUCCESS) cerr << "CUBLAS initialization error!\n";
    
    float alpha = 1.0f, beta = 0.0f;
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         cB, rA, cA,
                         &alpha, thrust::raw_pointer_cast(&dB[0]), cB,
                         thrust::raw_pointer_cast(&dA[0]), cA,
                         &beta,  thrust::raw_pointer_cast(&dC[0]), cB);
    if (status != CUBLAS_STATUS_SUCCESS) cerr << "Kernel execution error!\n";
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) cerr << "!!!! shutdown error (A)\n";
    vector<float> C(rA * cB);
    thrust::copy(dC.begin(), dC.end(), C.begin());
    t_end = clock();
    cout<<"CUBLAS Matrix Multiplication Time Usage:"<<endl;
    cout<< double(t_end - t_start)/CLOCKS_PER_SEC << " s"<<endl;
    cout<<endl;
    return C;
}
*/

template<>
void FastSim<gpu, double>::operator ()(const int &start_pos, const int &N_batch){
    assert(start_pos + N_batch <= N_samp);
    dvd dev_A = stgy, dev_B(N_feat * N_batch), dev_C(N_stgy * N_batch);
    for(int i=0;i<N_feat;++i) thrust::copy(signals[i].begin() + start_pos, signals[i].begin() + start_pos + N_batch, dev_B.begin() + i*N_batch);
    // Initialization of cuBlas
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if(status != CUBLAS_STATUS_SUCCESS) cerr << "CUBLAS initialization error!\n";
    
    double alpha = 1.0, beta = 0.0;
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N_batch, N_stgy, N_feat,
                         &alpha, thrust::raw_pointer_cast(&dev_B[0]), N_batch,
                         thrust::raw_pointer_cast(&dev_A[0]), N_feat,
                         &beta,  thrust::raw_pointer_cast(&dev_C[0]), N_batch);
    if (status != CUBLAS_STATUS_SUCCESS) cerr << "Kernel execution error!\n";
    // Finalization of cuBlas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) cerr << "!!!! shutdown error (A)\n";
    for(int i=0;i<N_stgy;++i){
        for(int j=0;j<N_batch;++j) cout<<dev_C[j]<<"\t";
        cout<<endl;
    }
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
        {0., 0., 1., 0.},
        {0., 0., 0., 1.}
    };
    FastSim<gpu, double> test(signals, prices);
    vector<int> late = {1, 1, 1, 1, 1, 1, 1, 1};
    auto res = test.getPerfectOps(late);
    cout<<"The optimal operation list is: \n";
    for(auto k:res) cout<<k<<' ';
    cout<<endl;
    cout<<"Testing loading weights function:"<<endl;
    test.loadWeights(weights);
    return 0;
}
