#include <memory.h>
#include <ctime>
#include <random>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <fstream>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <functional>
#include <queue>
#include <deque>
#include <stack>
#include <list>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>

#define REP(i,s,n) for(int (i)=s; (i)<(int)(n);(i)++)
#define RIT(it,c) for(__typeof(c.begin()) it = c.begin();it!=c.end();it++)
#define ALL(x) x.begin(), x.end()
#define SZ(x) (int)(x).size()
#define MSET(m,v) memset(m,v,sizeof(m))

using namespace std;

typedef long double ld;
typedef vector<int> vi;
typedef vector<long> vl;
typedef vector<bool> vb;
typedef vector<double> vd;
typedef pair<int,int> ii;
typedef pair<long, long> ll;
typedef unordered_set<int> ui;

class LinearReg{
    int N, K, N_test;
    vector<vd> train_X, test_X;
    vd train_Y, pred_Y, test_Y, weights;
    double bias;
public:
    LinearReg(vector<vd> train_x, vd train_y, vector<vd> test_x, vd test_y){
        train_X = train_x;
        train_Y = train_y;
        test_X = test_x;
        test_Y = test_y;
        
        // Make sure that all the dimensions are correct
        assert(!train_X.empty() && !train_X[0].empty());
        assert(!test_X.empty() && !test_X[0].empty());
        assert(train_X.size() == train_Y.size());
        assert(test_X.size() == test_Y.size());
        assert(train_X[0].size() == test_X[0].size());
        
        N = (int)train_X.size();
        K = (int)train_X[0].size();
        N_test = (int)test_X.size();
        
        weights = vd(K, 0.5);
        pred_Y.resize(N);
        bias = 0.5;
    }
    
    double getError(){
        // We use mean squre root error here.
        // Here we use the vector pred_Y to record the predicted value
        double sum = 0.;
        for(int i=0;i<N;++i) sum += pow(pred_Y[i] - train_Y[i], 2.);
        return sum/N;
    }
    
    double getTestError(){
        //This function is used to compute the Error for the test case.
        double sum = 0;
        for(int i=0;i<N_test;++i){
            double tmp = bias - test_Y[i];
            for(int j=0;j<K;++j) tmp += weights[j] * test_X[i][j];
            sum += pow(tmp, 2);
        }
        return sum/N;
    }
    
    void computePred(){
        for(int i=0;i<N;++i){
            pred_Y[i] = bias;
            for(int j=0;j<K;++j) pred_Y[i] += train_X[i][j] * weights[j];
        }
    }
    
    void oneStepUpdate(double learning_rate){
        // Using the simplest gradient descent
        for(int i=0;i<N;++i){
            bias -= learning_rate * (pred_Y[i] - train_Y[i])/N;
            for(int j=0;j<K;++j) weights[j] -= learning_rate * (pred_Y[i] - train_Y[i]) * train_X[i][j]/N;
        }
        return ;
    }
    
    double multipleSteps(int N_step, double learning_rate){
        computePred();
        for(int t=1; t<=N_step; ++t){
            oneStepUpdate(learning_rate);
            computePred();
        }
        return getError();
    }
    
    vd getWeights(){
        return weights;
    }
    
    double getBias(){
        return bias;
    }
};

class TestLinearReg{
    int n_var;
    double bias, Amplitude;
    vd weights;
    vector<vd> train_x, test_x;
    vd train_y, test_y;
    double linearFn(const vd &x){
        double ans = bias;
        for(int i=0;i<n_var;++i) ans += x[i]*weights[i];
        return ans;
    }
    double quardFn(const vd &x){
        assert(weights.size() == 2*x.size());
        double ans = bias;
        for(int i=0;i<n_var;++i){
            ans += weights[2*i]*x[i] + weights[2*i+1]*x[i]*x[i];
        }
        return ans;
    }
    double geneRand(){
        return double(rand())/RAND_MAX;
    }
public:
    TestLinearReg(double w1, double w2, double b, bool Quad = false, double Amp = 0.2){
        srand(1);
        weights = vd{w1, w2};
        bias = b;
        Amplitude = Amp;
        n_var = 2;
    }
    
    void generateDateSet(int N_train, int N_test){
        for(int i=0;i<N_train;++i){
            vd tmp_x = vd{1.7*geneRand(), 1.7*geneRand()};
            train_x.push_back(tmp_x);
            train_y.push_back(linearFn(tmp_x) + Amplitude * geneRand());
        }
        for(int i=0;i<N_test;++i){
            vd tmp_x = vd{1.7*geneRand(), 1.7*geneRand()};
            test_x.push_back(tmp_x);
            test_y.push_back(linearFn(tmp_x) + Amplitude * geneRand());
        }
    }
    
    void outputTrain(string filename){
        std::ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
        freopen(filename.c_str(), "w", stdout);
        for(int i=0;i<(int)train_x.size();++i){
            for(auto k:train_x[i]) cout<<k<<' ';
            cout<<train_y[i]<<endl;
        }
    }
    
    void outputTest(string filename){
        std::ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
        freopen(filename.c_str(), "w", stdout);
        for(int i=0;i<(int)test_x.size();++i){
            for(auto k:test_x[i]) cout<<k<<' ';
            cout<<test_y[i]<<endl;
        }
    }
    
    vector<vd> testModel(double l_rate,int n_block,int n_step){
        vector<vd> ans(3, vd());
        LinearReg model(train_x, train_y, test_x, test_y);
        double steps = 0.;
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
        double pred_bias = model.getBias();
        cerr<<"Here is what we obtain: y = x1*"<<pred_wei[0]<<" + x2*"<<pred_wei[1]<<" +"<<pred_bias<<endl;
        return ans;
    }
};

int main(int argc, char* argv[]){
    std::ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    double w1 = 1.7, w2 = 0.8, b = 2.2;
    int n_train = 7000, n_test = 3000;
    if(argc > 1) w1 = stod(argv[1]);
    if(argc > 2) w2 = stod(argv[2]);
    if(argc > 3) b = stod(argv[3]);
    if(argc > 4) n_train = stoi(argv[4]);
    if(argc > 5) n_test = stoi(argv[5]);
    TestLinearReg testLR(w1, w2, b);
    cerr<<"Going to fit this function: y = x1*"<<w1<<" + x2*"<<w2<<" +"<<b<<endl;
    cerr<<"Generating "<<n_train<<" training examples and "<<n_test<<" testing examples"<<endl;
    testLR.generateDateSet(n_train, n_test);
    string trainfile = "train_data.txt", testfile = "test_data.txt", resultfile = "serial_rslt.txt";
    testLR.outputTrain(trainfile);
    testLR.outputTest(testfile);
    cerr<<"Data sets are stored in "<<trainfile<<" and "<<testfile<<endl;
    cerr<<"Finish generating data"<<endl;
    cerr<<"Testing the model"<<endl;
    clock_t start_time = clock(), end_time;
    auto res = testLR.testModel(0.1, 100, 100);
    end_time = clock();
    float comp_time = float(end_time - start_time)/CLOCKS_PER_SEC;
    cerr<< setprecision(8);
    cerr<<"=========================================Time Usage========================================="<<endl<<endl;
    cout<<comp_time<<endl<<endl;
    cerr<<"============================================================================================"<<endl<<endl;
    cerr<<"Finish train the model"<<endl;
    freopen(resultfile.c_str(), "w", stdout);
    for(auto vec:res){
        for(auto k:vec) cout<<k<<' ';
        cout<<endl;
    }
    cerr<<"The cost function results are stored in "<<resultfile<<endl;
    return 0;
}
