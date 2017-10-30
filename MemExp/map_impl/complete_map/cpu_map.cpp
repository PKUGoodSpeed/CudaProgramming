/*
 We consider 1024 map, which are initially constructed with (at most)size = 4096 key, value pairs, for which keys are integers and values are float point numbers.
 Then each of them processes 262,144 operations. 
 The operations include:
 'i': insert key, value pair. If the map is full, or already have this element, do not insert, return 0.
 'r': remove a key. If the map do not have that element, do nothing and retion 0.
 'm': modify the value for a particular key (which can be combined with insert). If that value do not exist, return 0. and do nothing.
 's': search whether there is a key.
 'g': extract the value for a particular key. If did not find, return 0., but do not insert.
 'z': return the size of the map.
 'e': return whether the map is empty or not.
 'f': return whether the map is full or not.
 The above are symbols for the input operation list
 */
#include <bits/stdc++.h>
#include <cassert>

using namespace std;

const int NUM_INSTANCE = 1024;
const int NUM_OPERATION = 262144;
const int MAX_MAP_SIZE = 4096;
const int MOD = 10000;

// CPU version
class CPUMapTest{
    vector<map<int, float>> maps;
    vector<char> ops;
    vector<int> input_keys;
    vector<float> input_values;
    int N_ops, N_ins;
public:
    CPUMapTest(int num_ins): N_ins(num_ins){
        maps.resize(N_ins);
    }
    void loadOps(const vector<char> &ops, const vector<int> &input,const vector<float> &vals, int n_ops){
        N_ops = n_ops;
        assert((int)ops.size() == N_ops*N_ins);
        assert((int)input.size() == N_ops*N_ins);
        this->ops = ops;
        this->input_keys = input;
        this->input_values = vals;
        return;
    }
    void procOps(vector<float> &ans){
        ans.resize(N_ops * N_ins);
        for(int i=0;i<N_ins;++i){
            for(int j=0;j<N_ops;++j){
                char c = ops[i*N_ops + j];
                int key = input_keys[i*N_ops + j];
                float res = 0., value = input_values[i*N_ops + j];
                if(c == 'i'){
                    if((int)maps[i].size() < MAX_MAP_SIZE && !maps[i].count(key)){
                        maps[i][key] = value;
                        res = 1.;
                    }
                }
                else if(c == 'r'){
                    if(maps[i].count(key)){
                        maps[i].erase(key);
                        res = 1.;
                    }
                }
                else if(c == 'm'){
                    if(maps[i].count(key)) {
                        maps[i][key] = value;
                        res = 1.;
                    }
                }
                else if(c == 's') res = float(maps[i].count(key));
                else if(c == 'g'){
                    if(maps[i].count(key)) {
                        res = maps[i][key];
                    }
                }
                else if(c == 'z') res = float(maps[i].size());
                else if(c == 'e') res = float(maps[i].empty());
                else if(c == 'f') res = float((int)maps[i].size() == MAX_MAP_SIZE);
                else res = 0.;
                ans[i*N_ops + j] = res;
            }
        }
        return;
    }
};

int main(int argc, char *argv[]){
    srand(0);
    int num_ins = NUM_INSTANCE, num_ops = NUM_OPERATION;
    if(argc > 1) num_ins = stoi(argv[1]);
    if(argc > 2) num_ops = stoi(argv[2]);
    /* Generating data*/
    clock_t cpu_time = clock();
    string ref;
    ref += string(500, 'g') + string(500, 's') + string(200, 'i') + string(200, 'm') + "e" + "zz" + "f" + string(100, 'r');
    vector<char> ops(num_ins * num_ops);
    vector<int> input_keys(num_ins * num_ops);
    vector<float> input_values(num_ins * num_ops);
    generate(input_keys.begin(), input_keys.end(), [](){return rand()%MOD;});
    generate(input_values.begin(), input_values.end(), [](){return float(rand())/RAND_MAX;});
    generate(ops.begin(), ops.end(), [&ref](){return ref[rand()%(int)ref.size()];});
    cout<<"Time Usage for generating random data is: "<<float(clock() - cpu_time)/CLOCKS_PER_SEC<<"s"<<endl;
    

    cpu_time = clock();
    CPUMapTest cpu_test(num_ins);
    cpu_test.loadOps(ops, input_keys, input_values, num_ops);
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
