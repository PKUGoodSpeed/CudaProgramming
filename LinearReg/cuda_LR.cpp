#include <bits/stdc++.h>
using namespace std;

typedef vector<int> vi;
typedef vector<long> vl;
typedef vector<bool> vb;
typedef vector<float> vd;
typedef pair<int,int> ii;
typedef pair<long, long> ll;
typedef unordered_set<int> ui;

const int MAX_BLOCK_SIZE  = 1024;
const int MAX_NUM_FEATURES = 32;
const int MAX_CASE_PER_THREAD = 8;

__global__ void cudaUpdateWeight(int N, int K, int N_step, float l_rate, float *X, float *Y, float *new_w, float *old_w, int npt = 1){
    // Naive way to do updates
    assert(blockDim.x <= MAX_BLOCK_SIZE);
    assert(npt <= MAX_CASE_PER_THREAD);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float X_tmp[MAX_CASE_PER_THREAD][MAX_NUM_FEATURES], Y_pred[MAX_CASE_PER_THREAD], Y_true[MAX_CASE_PER_THREAD];
    int start = idx*npt, end = min(N, (idx+1)*npt);
    if(start >= N) return;
    for(int i=0;i<end-start;++i){
        Y_true[i] = Y[start + i];
        for(int j=0;j<K;++j){
            X_tmp[i][j] = X[(start + i)*K + j];
        }
    }
    for(int step = 0; step < N_step; ++step){
        for(int i=0;i<end-start;++i){
            Y_pred[i] = 0.;
            for(int j=0;j<K;++j) Y_pred[i] += X_tmp[i][j]*old_w[j];
        }
        for(int j = 0; j < K; ++j) {
            float additive = 0.;
            for(int i=0;i<end-start;++i) additive += (Y_true[i] - Y_pred[i])*X_tmp[i][j];
            additive *= l_rate/N;
            atomicAdd(&new_w[j], additive);
        }
        if(idx < K) old_w[idx] = new_w[idx];
    }
    return;
}

__global__ void cudaGetError(int N, int K, float *X, float *Y, float *weights, float *dev_err, int npt = 1){
    assert(blockDim.x <= MAX_BLOCK_SIZE);
    assert(npt <= MAX_CASE_PER_THREAD);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx*npt, end = min(N, (idx+1)*npt);
    if(start >= N) return;
    float blk_sum = 0.;
    for(int i=start; i<end ; ++i){
        float diff = -Y[i];
        for(int j=0;j<K;++j) diff += weights[j] * X[i*K + j];
        blk_sum += diff * diff;
    }
    atomicAdd(dev_err, blk_sum);
    return;
}




class CudaLinearReg{
    int N_train, N_test, N_feat;
    int B_size, N_block;
    // In this case, for convenient, consider weights and bias together.
    float **X_train, *Y_train, **X_test, *Y_test, *weights;
    float *dev_X, *dev_Y, *dev_w, *dev_old_w, *dev_err;
    inline float getRandNum(){ return float(rand())/RAND_MAX; }
public:
    CudaLinearReg(int n_trian, int n_test, int n_feat, int block_size = 0):N_train(n_trian), N_test(n_test), N_feat(n_feat){
        
        // Create serial storage for X_train
        X_train = new float* [N_train];
        X_train[0] = new float [N_train * N_feat];
        for(int i=1;i<N_train;++i) X_train[i] = X_train[i-1] + N_feat;
        
        // Create serial storage for Y_train
        Y_train = new float [N_train];
        
        // Create serial storage for X_test
        X_test = new float* [N_test];
        X_test[0] = new float [N_test * N_feat];
        for(int i=1;i<N_test;++i) X_test[i] = X_test[i-1] + N_feat;
        
        // Create serial storage for Y_test
        Y_test = new float [N_test];
        
        // Create serial storage for weights
        weights = new float [N_feat];
        
        // Create memory for X, Y, weights, old_weights on GPU device
        cudaMalloc((void **)&dev_X, N_train*N_feat*sizeof(float));
        cudaMalloc((void **)&dev_Y, N_train*sizeof(float));
        cudaMalloc((void **)&dev_w, N_feat*sizeof(float));
        cudaMalloc((void **)&dev_old_w, N_feat*sizeof(float));
        cudaMalloc((void **)&dev_err, 1*sizeof(float));
        
        if(block_size <= 0 || block_size>MAX_BLOCK_SIZE) B_size = min(MAX_BLOCK_SIZE, N_train);
        else B_size = block_size;
        N_block = (N_train + B_size - 1)/B_size;
    }
    
    void loadData(float **trX, float *trY, float **teX, float *teY){
        // Make sure that all the dimensions are correct
        memcpy(X_train[0], trX[0], N_train*N_feat*sizeof(float));
        memcpy(Y_train, trY, N_train*sizeof(float));
        memcpy(X_test[0], teX[0], N_test*N_feat*sizeof(float));
        memcpy(Y_test, teY, N_test*sizeof(float));
    }
    
    void setBlocks(int block_size){
        if(block_size <= 0 || block_size>MAX_BLOCK_SIZE) B_size = min(MAX_BLOCK_SIZE, N_train);
        else B_size = block_size;
        N_block = (N_train + B_size - 1)/B_size;
    }
    
    void initWeights(float amp_weight = 2.0){
        // Initializing weights
        for(int i=0;i<N_feat;++i) weights[i] = getRandNum();
    }
    
    void initGPU(){
        // Initializing CUDA memories
        cudaMemcpy(dev_X, X_train[0], N_train*N_feat*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Y, Y_train, N_train*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_w, weights, N_feat*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_old_w, weights, N_feat*sizeof(float), cudaMemcpyHostToDevice);
    }
    
    float getError(bool isTest = false, int npt = 1){
        // We use mean squre root error here.
        // Here we use the vector pred_Y to record the predicted value
        float error = 0;
        int N = N_train;
        cudaMemcpy(dev_err, &error, 1*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_w, weights, N_feat*sizeof(float), cudaMemcpyHostToDevice);
        if(isTest){
            N = N_test;
            cudaMemcpy(dev_X, X_test[0], N*N_feat*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_Y, Y_test, N*sizeof(float), cudaMemcpyHostToDevice);
        }
        else{
            cudaMemcpy(dev_X, X_train[0], N*N_feat*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_Y, Y_train, N*sizeof(float), cudaMemcpyHostToDevice);
        }
        cudaGetError<<<N_block, B_size>>>(N, N_feat, dev_X, dev_Y, dev_w, dev_err, npt);
        cudaMemcpy(&error, dev_err, 1*sizeof(float), cudaMemcpyDeviceToHost);
        return error/N;
    }
    
    void cudaNaiveTrain(int N_step, float learning_rate, int npt = 1){
        // Call the GPU update, which uses the Naive approach.
        initGPU();
        cudaUpdateWeight<<<N_block, B_size>>>(N_train, N_feat, N_step, learning_rate, dev_X, dev_Y, dev_w, dev_old_w, npt);
        cudaMemcpy(weights, dev_w, N_feat*sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    vd getWeights(){
        vd ans(N_feat);
        for(int i=0;i<N_feat;++i) ans[i] = weights[i];
        return vd;
    }
    
    ~CudaLinearReg(){
        delete train_X[0];
        delete train_X;
        delete train_Y;
        delete test_X[0];
        delete test_X;
        delete test_Y;
        delete weights;
        
        cudaFree(dev_X);
        cudaFree(dev_Y);
        cudaFree(dev_old_w);
        cudaFree(dev_w);
    }
};

class TestLinearReg{
    float ampli, **train_x, *train_y, **test_x, *test_y;
    int N_train, N_test, N_feat;
    vd weights;
    float linearFn(float *x){
        float ans = 0.;
        for(int i=0;i<N_feat;++i) ans += x[i] * weights[i];
        return ans;
    }
    /*
    float quardFn(const vd &x){
        assert(weights.size() == 2*x.size());
        for(int i=0;i<n_var;++i){
            ans += weights[2*i]*x[i] + weights[2*i+1]*x[i]*x[i];
        }
        return ans;
    }*/
    inline float getRandNum(){ return float(rand())/RAND_MAX; }
public:
    TestLinearReg(vd correct_w, int n_tr, int n_te, float Amp = 0.2): weights(correct_w), N_train(n_tr), N_test(n_te), ampli(Amp){
        srand(1);
        N_feat = (int)weights.size();
        assert(N_feat > 1);
        CudaLinearReg lg_test(N_train , N_test, N_feat);
        
        
        // Allocate memories
        train_x = new float* [N_train];
        train_x[0] = new float [N_train*N_feat];
        for(int i=1;i<N_train;++i) train_x[i] = train_x[i-1] + N_feat;
        train_y = new float [N_train];
        
        test_x = new float* [N_test];
        test_x[0] = new float [N_test*N_feat];
        for(int i=1;i<N_test;++i) test_x[i] = test_x[i-1] + N_feat;
        test_y = new float [N_test];
        
        // Show something on the screen:
        cerr << setprecision(6);
        cerr<<"We are testing the following function \n y = "<<weights[0];
        for(int i=1;i<N_feat;++i) cout<<" + "<<weights[i]<<"x"<<to_string(i);
        cout<<endl;
    }
    
    void generateDateSet(int N_train, int N_test){
        for(int i=0;i<N_train;++i){
            for(int j=0; j<N_feat; ++j){
                if(!j) train_x[i][j] = 1.;
                else train_x[i][j] = ampli*getRandNum();
            }
            train_y[i] = linearFn(train_x[i]);
        }
        for(int i=0;i<N_test;++i){
            for(int j=0; j<N_feat; ++j){
                if(!j) test_x[i][j] = 1.;
                else test_x[i][j] = ampli*getRandNum();
            }
            test_y[i] = linearFn(test_x[i]);
        }
    }
    
    void outputTrain(string filename){
        std::ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
        freopen(filename.c_str(), "w", stdout);
        for(int i=0;i<N_train;++i){
            for(int j=0;j<N_feat;++j) cout<<train_x[i][j]<<' ';
            cout<<train_y[i]<<endl;
        }
    }
    
    void outputTest(string filename){
        std::ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
        freopen(filename.c_str(), "w", stdout);
        for(int i=0;i<N_test;++i){
            for(int j=0;j<N_feat;++j) cout<<test_x[i][j]<<' ';
            cout<<test_y[i]<<endl;
        }
    }
    /*
    vector<vd> testModel(float l_rate,int n_block,int n_step){
        vector<vd> ans(3, vd());
        LinearReg model(train_x, train_y, test_x, test_y);
        float steps = 0.;
        ans[0].push_back(steps);
        model.computePred();
        ans[1].push_back(model.getError());
        ans[2].push_back(model.getTestError());
        for(int i=0;i<n_block;++i){
            steps += n_step;
            ans[0].push_back(model.multipleSteps(n_step, l_rate));
            ans[1].push_back(model.getTestError());
        }
        vd pred_wei = model.getWeights();
        float pred_bias = model.getBias();
        cerr<<"Here is what we obtain: y = x1*"<<pred_wei[0]<<" + x2*"<<pred_wei[1]<<" +"<<pred_bias<<endl;
        return ans;
    }*/
    ~TestLinearReg(){
        delete train_x[0];
        delete train_x;
        delete train_y;
        delete test_x[0];
        delete test_x;
        delete test_y;
    }
};

int main(int argc, char* argv[]){
    std::ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    float w1 = 1.7, w2 = 0.8, b = 2.2;
    int n_train = 7000, n_test = 3000;
    if(argc > 1) w1 = stod(argv[1]);
    if(argc > 2) w2 = stod(argv[2]);
    if(argc > 3) b = stod(argv[3]);
    if(argc > 4) n_train = stoi(argv[4]);
    if(argc > 5) n_test = stoi(argv[5]);
    TestLinearReg testLR(n_train, n_test, 3, vd{b, w1, w2});
    /*
    cerr<<"Going to fit this function: y = x1*"<<w1<<" + x2*"<<w2<<" +"<<b<<endl;
    cerr<<"Generating "<<n_train<<" training examples and "<<n_test<<" testing examples"<<endl;
    testLR.generateDateSet(n_train, n_test);
    string trainfile = "train_data.txt", testfile = "test_data.txt", resultfile = "serial_rslt.txt";
    testLR.outputTrain(trainfile);
    testLR.outputTest(testfile);
    cerr<<"Data sets are stored in "<<trainfile<<" and "<<testfile<<endl;
    cerr<<"Finish generating data"<<endl;
    cerr<<"Testing the model"<<endl;
    auto res = testLR.testModel(0.1, 100, 100);
    cerr<<"Finish train the model"<<endl;
    freopen(resultfile.c_str(), "w", stdout);
    for(auto vec:res){
        for(auto k:vec) cout<<k<<' ';
        cout<<endl;
    }
    cerr<<"The cost function results are stored in "<<resultfile<<endl;*/
    return 0;
}
