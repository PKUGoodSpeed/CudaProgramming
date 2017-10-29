/*
 We consider 1024 map, which are initially constructed with (at most)size = 4096 key, value pairs, for which keys are integers and values are float point numbers.
 Then each of them processes 1,048,576 operations, which only include 's': search, 'e': check empty, 'z': check size and 'g': getting values.
 */
#include <bits/stdc++.h>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#define to_ptr(x) thrust::raw_pointer_cast(&x[0])
#define gpu_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define def_dvec(t) thrust::device_vector<t>

using namespace std;

const int N_inst = 1024;
const int N_ops = 1048576;
const int MAX_MAP_SIZE = 4096;

// CPU version
class CPUMapTest{
    vector<map<int, float>> maps;
    vector<char> ops;
    vector<int> input;
    int N_ops, N_ins;
public:
    CPUMapTest(const vector<vector<int>> &keys, const vector<vector<float>> &values) N_ins((int)keys.size()){
        maps.resize(keys.size());
        for(int i=0;i<(int)keys.size();++i){
            for(int j=0;j<keys[i].size();++j) maps[i][keys[i]] = values[i];
        }
    }
    void loadOps(const vector<char> &ops, const vector<int> &input){
        assert((int)ops.size() == N_ops*N_ins)
        this->ops = ops;
        this->input = input;
    }
};
