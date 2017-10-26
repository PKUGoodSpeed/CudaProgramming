#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <fstream>
#include <cassert>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cublas_v2.h>

#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))

using namespace std;

const int BLOCK_SIZE = 512;

__global__ void simKernel(int N_stgy, int N_batch, int from, int n_ts, float *alpha, float *mid, float *gap, int *late, int *pos, int *rest_lag, float *prof, float *last_prc, int *cnt, float fee){
    int global_i = blockIdx.x*blockDim.x + threadIdx.x;
    if( global_i >= N_stgy) return;
    int start = global_i*N_batch + rest_lag[global_i], end = global_i*N_batch + n_ts, i;
    for(i = start; i<end; ++i) if(alpha[i]*mid[i-start+from]>gap[i-start+from] + fee || alpha[i]*mid[i-start+from]<-gap[i-start+from] - fee){
        if(alpha[i]*mid[i-start+from]>gap[i-start+from]+fee && pos[global_i]<1){
            last_prc[global_i] = mid[i-start+from] + gap[i-start+from] + fee;
            prof[global_i] -= (1-pos[global_i])*last_prc[global_i];
            cnt[global_i] += 1-pos[global_i];
            pos[global_i] = 1;
            i += late[i-start+from];
        }
        else if(alpha[i]*mid[i-start+from]<-gap[i-start+from]-fee && pos[global_i]>-1){
            last_prc[global_i] = mid[i-start+from] - gap[i-start+from] - fee;
            prof[global_i] += (pos[global_i]+1)*last_prc[global_i];
            cnt[global_i] += pos[global_i]+1;
            pos[global_i] = -1;
            i += late[i-start+from];
        }
    }
    rest_lag[global_i] = i-end;
}

template <typename D_TYPE>
class GPUFastSim{
    typedef vector<D_TYPE> vec_t;
    typedef vector<int> vi;
    typedef thrust::device_vector<D_TYPE> dvec_t;
    typedef thrust::device_vector<int> dvi;
    
    int N_samp, N_feat, N_stgy, N_batch;
    dvec_t mid, gap, weights, dev_feats, dev_logi;
    
    // Here only feature/signal values are stored on CPU
    vector<vec_t> signals;
    // The following vectores are used to mark the status of each simulation trajectory
    dvec_t prof, last_prc;
    dvi pos, rest_lag, trd_cnt;
    // The following vector latencies is used to store the latency information
    dvi late;
public:
    FastSim(const vector<vec_t> &sigs, const vector<vec_t> &prices, const int &n_batch):N_stgy(0), signals(sigs), N_batch(n_batch){
        assert(!sigs.empty() && !sigs[0].empty() && !prices.empty() && (int)prices.size() == 2);
        assert(sigs[0].size() == prices[0].size());
        N_samp = (int)sigs[0].size();
        N_feat = (int)sigs.size();
        mid.resize(N_samp);
        gap.resize(N_samp);
        vi tmp_mid(N_samp), tmp_gap(N_samp);
        transform(prices[0].begin(), prices[0].end(), prices[1].begin(), tmp_mid.begin(), [](DATA_TYPE x, DATA_TYPE y){return abs(y+x)/2;});
        transform(prices[0].begin(), prices[0].end(), prices[1].begin(), tmp_gap.begin(), [](DATA_TYPE x, DATA_TYPE y){return abs(y-x)/2;});
        gpu_copy(tmp_mid, mid);
        gpu_copy(tmp_gap, gap);
        late.assign(N_samp, 0);
        dev_feats.resize(N_feat * N_batch);
    }
    /* loading the strategy weights into this objects */
    void loadWeights(const vector<vec_t> &weights){
        assert(!weights.empty() && !weights[0].empty());
        assert((int)weights[0].size() == N_feat);
        N_stgy = (int)weights.size();
        stgy.resize(N_stgy * N_feat);
        for(int i=0;i<N_stgy;++i) gpu_copy_to(weights[i], stgy, i*N_feat);
        dev_logi.resize(N_stgy * N_batch);
        rest_lag = pos = trd_cnt = dvi(N_stgy, 0);
        last_prc = prof = dvec_t(N_stgy, 0.);
    }
    
    /* Place holder */
    void loadLatencies(const vector<int> &late){}
    
    /* Do gpu fastSim */
    void operator ()(const int &start_pos, D_TYPE fee){
        int end_pos = min(start_pos + N_batch, N_samp);
        clock_t t_start, t_end;
        t_start = clock();
        for(int i=0;i<N_feat;++i) thrust::copy(signals[i].begin() + start_pos, signals[i].begin() + end_pos, dev_feats.begin() + i*N_batch);
        
        // Initialization of cuBlas
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if(status != CUBLAS_STATUS_SUCCESS) cerr << "CUBLAS initialization error!\n";
        
        float alpha = 1.0, beta = 0.0;
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N_batch, N_stgy, N_feat,
                             &alpha, to_ptr(dev_feats), N_batch,
                             to_ptr(stgy), N_feat,
                             &beta,  to_ptr(dev_logi), N_batch);
        if (status != CUBLAS_STATUS_SUCCESS) cerr << "Kernel execution error!\n";
        // Finalization of cuBlas
        status = cublasDestroy(handle);
        if (status != CUBLAS_STATUS_SUCCESS) cerr << "!!!! shutdown error (A)\n";
        t_end = clock();
        cout<<"Time usage for matrix multiplication is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl;
        
        t_start = clock();
        
        // Doing parallelized fast simulation
        simKernel<<<(N_stgy + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(N_stgy, N_batch, start_pos, end_pos-start_pos, to_ptr(dev_logi), to_ptr(mid), to_ptr(gap),
                                                                        to_ptr(late), to_ptr(pos), to_ptr(rest_lag), to_ptr(prof), to_ptr(last_prc), to_ptr(cnt), fee);
        // Copy status to CPU
        t_end = clock();
        cout<<"Time usage for running simulation is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl;
        return;
    }
    void showResults(){
        cout<<"Showing the Results:"<<endl;
        for(int i=0;i<N_stgy;i+=N_stgy/12+1){
            for(int j=0;j<N_feat;++j) cout<<stgy[i*N_feat + j]<<' ';
            cout<<prof[i]<<' '<<trd_cnt[i]<<endl;
        }
        return;
    }
    /*
    void finalizeSim(){
        transform(pos.begin(), pos.end(), last_prc.begin(), last_prc.begin(), [](int x, DATA_TYPE y){return (DATA_TYPE)x*y;});
        transform(prof.begin(), prof.end(), last_prc.begin(), prof.begin(), plus<DATA_TYPE>());
        transform(trd_cnt.begin(), trd_cnt.end(), trd_cnt.begin(), [](int x){return max(0, x-1);});
        cout<<"Showing the Results:"<<endl;
        for(int i=0;i<N_stgy;i+=N_stgy/12+1){
            for(int j=0;j<N_feat;++j) cout<<stgy[i*N_feat + j]<<' ';
            cout<<prof[i]<<' '<<trd_cnt[i]<<endl;
        }
        return;
    }*/
    void fastSimulation(const D_TYPE &fee){
        clock_t t_start = clock()
        for(int i=0;i<N_samp;i+=N_batch) this->operator()(i, fee);
        clock_t t_end = clock();
        cout<<"Time usage for whole gpu fast sim is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl;
        this->showResults();
        return;
    }
};

int main(int argc, char *argv[]){
    assert(argc > 1);
    ifstream fin;
    fin.open(argv[1]);
    int N_samp, N_feat = 3, N_stgy = 1000;
    fin>>N_samp;
    cout<<"The number of timestamps is "<<N_samp<<endl;
    clock_t t_start = clock();
    vector<vector<float>> prices(2, vector<float>(N_samp)), signals(N_feat, vector<float>(N_samp));
    for(int i=0;i<N_samp;++i){
        for(int j=0;j<3;++j) fin>>signals[j][i];
        for(int j=0;j<2;++j) fin>>prices[1-j][i];
    }
    FastSim<float> test(signals, prices, 200000);
    clock_t t_end = clock();
    cout<<"Time usage for reading the data is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    
    cout<<"Testing GPU fast sim performance:\n";
    cout<<"Generating weights"<<endl;
    t_start = clock();
    vector<vector<float>> weights(N_stgy, vector<float>(N_feat));
    for(int i=0;i<N_stgy;++i){
        for(int j=0, m=i;j<N_feat; ++j){
            weights[i][j] = (0.1*(m%10) + 0.05)*3.;
            m /= 10;
        }
    }
    t_end = clock();
    cout<<"Time usage for generating the weights is "<<double(t_end - t_start)/CLOCKS_PER_SEC<<" s"<<endl<<endl;
    test.loadWeights(weights);
    test.fastSimulation(double(0.008));
    return 0;
}

