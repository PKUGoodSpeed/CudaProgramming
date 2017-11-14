#undef _GLIBCXX_USE_INT128

#include <bits/stdc++.h>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cublas_v2.h>

#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define gpu_copy_to(x, y, pos) thrust::copy((x).begin(), (x).end(), (y).begin() + (pos))
#define def_dvec(t) thrust::device_vector<t>

using namespace std;
const int BLOCK_SIZE = 1024;

__global__ void cudaGetShiftedMidPrice(int N_inst, int batch_size, float *alphas, float *mid, float *shifted_prc){
    int b_sz = blockDim.x, b_id = blockIdx.x, t_id = threadIdx.x;
    if(b_id < N_inst){
        for(int i=t_id; i<batch_size; i += b_sz){
            shifted_prc[b_id * batch_size + i] = (1. + alphas[b_id * batch_size + i]) * mid[i];
        }
    }
}

class PredictMidPrice{
    def_dvec(float) dev_weights, dev_feats, dev_alphas, dev_shifted_prc, dev_mid;
    vector<vector<float>> market_feats, shifted_mid_prc;
    vector<float> mid;
    int N_samp, N_inst, N_feat;
    cudaEvent_t event_start, event_stop; // Using cuda events to evaluate time
    float cuda_time;
public:  
    PredictMidPrice(const int &num_timestamp, const int &num_instance,
    const int &num_feature, const vector<vector<float>> &features,
    const vector<vector<float>> &stgy_weights, const vector<float> & mid_prc): 
    N_samp(num_timestamp), N_inst(num_instance), N_feat(num_feature), 
    market_feats(features), mid(mid_prc){
        assert(N_samp && N_inst && N_feat);
        assert((int)market_feats.size() == N_feat);
        assert((int)market_feats[0].size() == N_samp);
        assert((int)stgy_weights.size() == N_inst);
        assert((int)stgy_weights[0].size() == N_feat);
        assert((int)mid.size() == N_samp);
        
        /* Initialize the output vector */
        shifted_mid_prc = vector<vector<float>>(N_inst,vector<float>(N_samp, 0.));
        
        /* Initialize weights information */
        dev_weights.resize(N_inst * N_feat);
        for(int i=0;i<N_inst;++i){
            gpu_copy_to(stgy_weights[i], dev_weights, i*N_feat);
        }
        
        /* Create events for measuring time cost */
        cudaEventCreate(&event_start);   // creating the event 1
        cudaEventCreate(&event_stop);    // creating the event 2
    }
    void operator ()(int start_pos, int b_size){
        cudaEventRecord(event_start, 0);        // Record the starting event
        int end_pos = min(N_samp, start_pos + b_size);
        for(int i=0;i<N_feat;++i) {
            thrust::copy(market_feats[i].begin() + start_pos, 
            market_feats[i].begin() + end_pos, dev_feats.begin() + i*b_size);
        }
        /* Matrix multiplication section */
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if(status != CUBLAS_STATUS_SUCCESS) cerr << "CUBLAS initialization error!\n";
    
        float alpha = 1.0, beta = 0.0;
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            b_size, N_inst, N_feat,
                            &alpha, to_ptr(dev_feats), b_size,
                            to_ptr(dev_weights), N_feat,
                            &beta,  to_ptr(dev_alphas), b_size);
        if (status != CUBLAS_STATUS_SUCCESS) cerr << "Kernel execution error!\n";
        // Finalization of cuBlas
        status = cublasDestroy(handle);
        if (status != CUBLAS_STATUS_SUCCESS) cerr << "!!!! shutdown error (A)\n";
        cudaEventRecord(event_stop, 0);                  // Record the ending event
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&cuda_time, event_start, event_stop); // Measure the time between 
        cout<<"Time Usage for computing alphas is: "<<cuda_time/1000<<"s"<<endl;
        
        /* Computing the predicted shifted mid prices */
        cudaEventRecord(event_start, 0);        // Record the starting event
        thrust::copy(mid.begin() + start_pos, mid.begin() + end_pos, dev_mid.begin());
        cudaGetShiftedMidPrice<<<N_inst, BLOCK_SIZE>>>(N_inst, b_size, to_ptr(dev_alphas), 
        to_ptr(dev_mid), to_ptr(dev_shifted_prc));
        for(int i=0;i<N_inst;++i){
            thrust::copy(dev_shifted_prc.begin() + i*b_size, 
            dev_shifted_prc.begin() + i*b_size + end_pos - start_pos, 
            shifted_mid_prc[i].begin() + start_pos);
        }
        cudaEventRecord(event_stop, 0);                  // Record the ending event
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&cuda_time, event_start, event_stop); // Measure the time between 
        cout<<"Time Usage for computing shifted_mid_prc is: "<<cuda_time/1000<<"s"<<endl;
    }
    
    void computShiftedPrice(const int &batch_size){
        dev_feats.resize(N_feat * batch_size);
        dev_alphas.resize(N_inst * batch_size);
        dev_mid.resize(batch_size);
        dev_shifted_prc.resize(N_inst * batch_size);
        for(int i=0;i<N_samp;i+=batch_size){
            this->operator ()(i, batch_size);
        }
    }
    
    vector<vector<float>> getShiftedPrice(){
        return shifted_mid_prc;
    }
};

class LevelBookSize{
    const float delta = 1.E-6;
    float tick_size;
    int base_level, num_level, N_samp;
    vector<int> ask, bid;
    vector<vector<int>> a_size, b_size;
public:
    LevelBookSize(const float &ts, const vector<float> &ap, 
    const vector<vector<float>> &asz, const vector<float> &bp,
    const vector<vector<float>> &bsz): tick_size(ts){
        N_samp = (int)ap.size();
        assert((int)bp.size() == N_samp);
        assert((int)asz.size() == N_samp);
        assert((int)asz[0].size() == 5);
        assert((int)bsz.size() == N_samp);
        assert((int)bsz[0].size() == 5);
        ask.resize(N_samp);
        bid.resize(N_samp);
        a_size = vector<vector<int>>(N_samp, vector<int>(5, 0));
        b_size = vector<vector<int>>(N_samp, vector<int>(5, 0));
        int Max_level = 0, Min_level = 1E8;
        for(int i=0;i<N_samp;++i){
            ask[i] = int((ap[i] + delta)/tick_size);
            bid[i] = int((bp[i] + delta)/tick_size);
            Max_level = max(Max_level, ask[i] + 10);
            Min_level = min(Min_level, bid[i] - 10);
            for(int j=0;j<5;++j) a_size[i][j] = int(asz[i][j] + delta);
            for(int j=0;j<5;++j) b_size[i][j] = int(bsz[i][j] + delta);
        }
        base_level = Min_level;
        num_level = Max_level - Min_level + 1;
    }
    int getBaseLevel(){
        return base_level;
    }
    int getNumberLevels(){
        return num_level;
    }
    vector<vector<int>> getAskSizeBook(){
        vector<vector<int>> ask_size_book(N_samp, vector<int>(num_level, 0));
        for(int i=0;i<N_samp;++i){
            for(int j=0;j<5;++j){
                ask_size_book[i][ask[i] - base_level + j] = a_size[i][j];
            }
        }
        return ask_size_book;
    }
    vector<vector<int>> getBidSizeBook(){
        vector<vector<int>> bid_size_book(N_samp, vector<int>(num_level, 0));
        for(int i=0;i<N_samp;++i){
            for(int j=0;j<5;++j){
                bid_size_book[i][bid[i] - base_level - j] = b_size[i][j];
            }
        }
        return bid_size_book;
    }
};

int main(){
    cout<< "Test#1" <<endl;
    vector<float> mid{1., 2., 1., 2., 1., 2., 1., 2., 1.};
    vector<vector<float>> features(3, vector<float>(9, 1.));
    vector<vector<float>> weights(10, vector<float>(3, 0.2));
    PredictMidPrice symp_test(9, 10, 3, features, weights, mid);
    symp_test.computShiftedPrice(3);
    auto ans = symp_test.getShiftedPrice();
    for(auto vec: ans){
        for(auto k:vec) cout<<k<<' ';
        cout<<endl;
    }
    
    cout<< endl << "Test#2" <<endl;
    vector<float> bid{2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9};
    vector<float> ask{2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0};
    vector<vector<float>> az(10, vector<float>(5,1.)), bz(10, vector<float>(5,1.));
    for(int i=0;i<10;++i){
        for(int j=0;j<5;++j) az[i][j] = bz[i][j] = 5.+j;
    }
    LevelBookSize book_test(0.1, ask, az, bid, bz);
    auto base_level= book_test.getBaseLevel();
    auto num_level = book_test.getNumberLevels();
    cout<< base_level << ' ' << num_level <<endl;
    auto askbz= book_test.getAskSizeBook(), bidbz = book_test.getBidSizeBook();
    cout<< "ASK BOOK SIZE" <<endl;
    for(int j=num_level-1;j>=0;--j){
        for(int i=0;i<10;++i) cout<<askbz[i][j]<<' ';
        cout<<endl;
    }
    cout<< "BID BOOK SIZE" <<endl;
    for(int j=num_level-1;j>=0;--j){
        for(int i=0;i<10;++i) cout<<bidbz[i][j]<<' ';
        cout<<endl;
    }
    return 0;
}