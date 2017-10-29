/*
 We consider 1024 map, which are initially constructed with (at most)size = 4096 key, value pairs, for which keys are integers and values are float point numbers.
 Then each of them processes 1,048,576 operations, which only include 'g': search/getting values (If did not find, return -1.), 'e': check empty, 'z': check size.
 */
#include <bits/stdc++.h>
#include <cassert>

using namespace std;

const int NUM_INSTANCE = 1024;
const int NUM_OPERATION = 1048576;
const int MAX_MAP_SIZE = 4096;
const int MOD = 1000;

// CPU version
class CPUMapTest{
    vector<map<int, float>> maps;
    vector<char> ops;
    vector<int> input;
    int N_ops, N_ins;
public:
    CPUMapTest(const vector<vector<int>> &keys, const vector<vector<float>> &values): N_ins((int)keys.size()){
        maps.resize(keys.size());
        for(int i=0;i<(int)keys.size();++i){
            for(int j=0;j<(int)keys[i].size();++j) maps[i][keys[i][j]] = values[i][j];
        }
    }
    void loadOps(const vector<char> &ops, const vector<int> &input, int n_ops){
        N_ops = n_ops;
        assert((int)ops.size() == N_ops*N_ins);
        assert((int)input.size() == N_ops*N_ins);
        this->ops = ops;
        this->input = input;
        return;
    }
    void procOps(vector<float> &ans){
        ans.resize(N_ops * N_ins);
        for(int i=0;i<N_ins;++i){
            for(int j=0;j<N_ops;++j){
                if(ops[i*N_ops + j] == 'e'){
                    ans[i*N_ops + j] = float(maps[i].empty());
                }
                else if(ops[i*N_ops + j] == 'z'){
                    ans[i*N_ops + j] = float(maps[i].size());
                }
                else{
                    if(maps[i].count(input[i*N_ops + j])) ans[i*N_ops + j] =  maps[i][input[i*N_ops + j]];
                    else ans[i*N_ops + j] = -1.;
                }
            }
        }
        return;
    }
};

int main(int argc, char *argv[]){
    int size = MAX_MAP_SIZE, num_ins = NUM_INSTANCE, num_ops = NUM_OPERATION;
    if(argc > 1) num_ins = stoi(argv[1]);
    if(argc > 2) num_ops = stoi(argv[2]);
    /* Generating data*/
    clock_t cpu_time = clock();
    vector<vector<int>> keys(num_ins);
    vector<vector<float>> values(num_ins);
    vector<char> ops(num_ins * num_ops, 'g');
    vector<int> input(num_ins * num_ops);
    for(int i=0;i<num_ins;++i){
        unordered_set<int> exist;
        for(int j=0;j<size;++j){
            int tmp_key = rand()%MOD;
            if(!exist.count(tmp_key)){
                keys[i].push_back(tmp_key);
                values[i].push_back(float(rand())/RAND_MAX);
                exist.insert(tmp_key);
            }
        }
    }
    generate(input.begin(), input.end(), [](){return rand()%MOD;});
    for(int i=0;i<num_ins;++i){
        ops[i*num_ops] = 'z';
        ops[i*num_ops + 1] = 'e';
        
    }
    cout<<"Time Usage for generating random data is: "<<float(clock() - cpu_time)/CLOCKS_PER_SEC<<"s"<<endl;
    
    cpu_time = clock();
    CPUMapTest cpu_test(keys, values);
    cpu_test.loadOps(ops, input, num_ops);
    cout<<"Time Usage for preparing maps is: "<<float(clock() - cpu_time)/CLOCKS_PER_SEC<<"s"<<endl;
    
    cpu_time = clock();
    vector<float> ans;
    cpu_test.procOps(ans);
    cout<<"Time Usage for processing operations is: "<<float(clock() - cpu_time)/CLOCKS_PER_SEC<<"s"<<endl;
    
    cout<<"Showing CPU code answers:"<<endl;
    for(int i=0;i<num_ins*num_ops ; i+=num_ins*num_ops/25 + 1) cout<<ans[i]<<' ';
    cout << endl;
    cout<<"DONE!"<<endl;
    
    return 0;
}
