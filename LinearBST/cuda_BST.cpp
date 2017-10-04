#include <bits/stdc++.h>
using namespace std;

/*
 * Here I try to use one-dimensional arrays to realize a Binary Searching Tree, such that a tree can be built on the memory of GPU threads.
 * Since we finally want to model map on threads, so the thing stored on each tree node is (key,value> pairs.
 * For simplicity, we only consider interger keys and values. (For float key, we need to add a small margin for the operator "==".)
 * We test the model by inputing a operation list, including insertion: 'i', searching: 's', getting value: 'g', removal: 'r', checking empty: 'e', checking full: 'f', and getting size: 'z'
 * The disadvantage of using GPU is that once the memory automatically release the memory if it gets out of the kernel function.
 */

const int MAX_SIZE = 4096;
const int MAX_BLOCK_SIZE = 1024;
const int MAX_NUM_BLOCK = 1024;
const int MAX_DATA = 20000;

// The following three are most simply functions: as their name describes
__device__ int isEmpty(int rest){
    return (rest == MAX_SIZE - 1);
}

__device__ void isFull(int *full, int rest){
    (*full) = (!rest);
    return;
}
__device__ void getSize(int *size, int rest){
    (* size) = (MAX_SIZE - rest - 1);
    return;
}


// The following function for searching a key in the item: found: return true; not found: return false.
__device__ void searchItem(int *exist, int rest, int root, int *keys, int *children, int key){
    (*exist) = 0;
    if(rest == MAX_SIZE - 1) return;
    int p = root;
    while(key != keys[p]){
        if(key < keys[p]){
            if(!(children[p]/MAX_SIZE)) return;
            p = children[p]/MAX_SIZE;
        }
        else{
            if(!(children[p]%MAX_SIZE)) return;
            p = children[p]%MAX_SIZE;
        }
    }
    (*exist) = 1;
    return;
}

// The following function for getting the value for an input key (Here we do not insert the key if the key does not exist previously)
__device__ void getItem(int *value, int rest, int root, int *keys, int *values, int *children, int key){
    assert(rest < MAX_SIZE-1);
    int p = root;
    while(key != keys[p]){
        if(key < keys[p]){
            assert(children[p]/MAX_SIZE);
            p = children[p]/MAX_SIZE;
        }
        else{
            assert(children[p]%MAX_SIZE);
            p = children[p]%MAX_SIZE;
        }
    }
    (*value) = values[p];
    return;
}

// The following function is used for inserting a key-value pair into the tree.
__device__ void insertItem(int *rest, int *root, int *rest_idx, int *keys, int *values, int *parent, int *children, int key, int value){
    if((*rest) == MAX_SIZE - 1){
        (*rest)--;
        (*root) = rest_idx[(*rest)];
        keys[(*root)] = key;
        values[(*root)] = value;
        parent[(*root)] = 0;
        children[0] = (*root);
        return;
    }
    int p = (*root);
    while(key != keys[p]){
        if(key < keys[p]){
            if(!(children[p]/MAX_SIZE)){
                (*rest)--;
                int idx = rest_idx[(*rest)];
                keys[idx] = key;
                values[idx] = value;
                parent[idx] = p;
                children[p] += idx*MAX_SIZE;
                return;
            }
            p = children[p]/MAX_SIZE;
        }
        else{
            if(!(children[p]%MAX_SIZE)){
                (*rest)--;
                int idx = rest_idx[(*rest)];
                keys[idx] = key;
                values[idx] = value;
                parent[idx] = p;
                children[p] += idx;
                return;
            }
            p = children[p]%MAX_SIZE;
        }
    }
    values[p] = value;
    return;
}

// The following function is used to remove a node from the tree.
__device__ void removeItem(int *rest, int *root, int *rest_idx, int *keys, int *parent, int *children, int key){
    if((*rest) == MAX_SIZE - 1) return;
    int p = (*root);
    while(key != keys[p]){
        if(key < keys[p]){
            if(!(children[p]/MAX_SIZE)) return;
            p = children[p]/MAX_SIZE;
        }
        else{
            if(!(children[p]%MAX_SIZE)) return;
            p = children[p]%MAX_SIZE;
        }
    }
    int par = parent[p], cur = p;
    if(!(children[p]/MAX_SIZE)) cur = children[p]%MAX_SIZE;
    else if(!(children[children[p]/MAX_SIZE]%MAX_SIZE)){
        cur = children[p]/MAX_SIZE;
        children[cur] += children[p]%MAX_SIZE;
        parent[children[p]%MAX_SIZE] = cur;
    }
    else{
        cur = children[children[p]/MAX_SIZE]%MAX_SIZE;
        while(children[cur]%MAX_SIZE) cur = children[cur]%MAX_SIZE;
        children[parent[cur]] -= cur;
        children[cur] = children[p];
        parent[children[p]/MAX_SIZE] = cur;
        parent[children[p]%MAX_SIZE] = cur;
    }
    if(children[par]/MAX_SIZE == p) children[par] += (cur - p)*MAX_SIZE;
    else children[par] += cur - p;
    parent[cur] = par;
    if(p == (*root)) (*root) = cur;
    rest_idx[(*rest)] = p;
    (*rest)++;
    return;
}

__global__ void procOperations(int N_threads, int *rest, int *root, int *rest_idx, int *keys, int *values, int *parent, int *children, int N_op, char *ops, int *inputs,int *ans, int *sync){
    int cnt = 0, g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for(int i=0;i<N_op;++i){
        if(cnt >= N_threads) cnt -= N_threads;
        if(ops[i] == 'e' && g_idx == cnt){
            ans[i] = isEmpty((*rest));
            cnt++;
        }
        else if(ops[i] == 'f' && g_idx == cnt){
            isFull(&ans[i], (*rest));
            cnt++;
        }
        else if(ops[i] == 'z' && g_idx == cnt){
            getSize(&ans[i], (*rest));
            cnt++;
        }
        else if(ops[i] == 's' && g_idx == cnt){
            searchItem(&ans[i], (*rest), (*root), keys, children, inputs[i]);
            cnt++;
        }
        else if(ops[i] == 'g' && g_idx == cnt){
            getItem(&ans[i], (*rest), (*root), keys, values, children, inputs[i]);
        }
        else if(ops[i] == 'r'){
            atomicAdd(sync, 0);
            if(g_idx == 0) removeItem(rest, root, rest_idx, keys, parent, children, inputs[i]);
            atomicAdd(sync, 0);
            cnt = 0;
        }
        else{
            atomicAdd(sync, 0);
            if(g_idx == 0) insertItem(rest, root, rest_idx, keys, values, parent, children, inputs[i]/MAX_DATA, inputs[i]%MAX_DATA);
            atomicAdd(sync, 0);
            cnt = 0;
        }
    }
    return;
}

class CudaBST{
    int root, rest, *rest_idx, *keys, *values, *parent, *children;
    int *dev_root, *dev_rest, *dev_rest_idx, *dev_keys, *dev_values, *dev_parent, *dev_children, *sync;
    int block_size, num_blocks, num_threads;
    int *dev_inputs, N_ops, *dev_ans;
    char *dev_ops;
public:
    CudaBST(int b_size, int n_blocks){
        assert(b_size <= MAX_BLOCK_SIZE);
        assert(n_blocks <= MAX_NUM_BLOCK);
        block_size = b_size;
        num_blocks = n_blocks;
        num_threads = num_blocks * block_size;
        
        // Assign memories for CPU variables
        rest_idx = new int [MAX_SIZE];
        keys = new int [MAX_SIZE];
        values = new int [MAX_SIZE];
        parent = new int [MAX_SIZE];
        children = new int [MAX_SIZE];
        
        
        // Assign memories for GPU variables
        cudaMalloc((void **)&dev_root, 1*sizeof(int));
        cudaMalloc((void **)&dev_rest, 1*sizeof(int));
        cudaMalloc((void **)&sync, 1*sizeof(int));
        cudaMalloc((void **)&dev_rest_idx, MAX_SIZE*sizeof(int));
        cudaMalloc((void **)&dev_keys, MAX_SIZE*sizeof(int));
        cudaMalloc((void **)&dev_values, MAX_SIZE*sizeof(int));
        cudaMalloc((void **)&dev_parent, MAX_SIZE*sizeof(int));
        cudaMalloc((void **)&dev_children, MAX_SIZE*sizeof(int));
    }
    
    // Initiating the BST
    void init(){
        root = 0;
        rest = MAX_SIZE - 1;
        for(int i=0;i<MAX_SIZE;++i) rest_idx[i] = MAX_SIZE - 1 - i;
    }
    
    // Transfer operation list onto GPU
    void loadOperations(int n_ops, char *ops, int *inputs){
        N_ops = n_ops;
        cudaMalloc((void **)&dev_ops, N_ops*sizeof(char));
        cudaMemcpy(dev_ops, ops, N_ops*sizeof(char), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&dev_inputs, N_ops*sizeof(int));
        cudaMemcpy(dev_inputs, inputs, N_ops*sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&dev_ans, N_ops*sizeof(int));
    }
    
    // Processing the operation list
    void runOperations(int *ans){
        cudaMemcpy(dev_ans, ans, N_ops*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_rest, &rest, 1*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_root, &root, 1*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_rest_idx, rest_idx, MAX_SIZE*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_keys, keys, MAX_SIZE*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_values, values, MAX_SIZE*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_parent, parent, MAX_SIZE*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_children, children, MAX_SIZE*sizeof(int), cudaMemcpyHostToDevice);
        procOperations<<<num_blocks, block_size>>>(num_threads, dev_rest, dev_root, dev_rest_idx, dev_keys, dev_values, dev_parent, dev_children, N_ops, dev_ops, dev_inputs, dev_ans, sync);
        cudaMemcpy(&rest, dev_rest, 1*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&root, dev_root, 1*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(rest_idx, dev_rest_idx, MAX_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(keys, dev_keys, MAX_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(values, dev_values, MAX_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(parent, dev_parent, MAX_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(children, dev_children, MAX_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(ans, dev_ans, N_ops*sizeof(int), cudaMemcpyDeviceToHost);
        for(int i=0;i<N_ops;++i)cout<<ans[i];
    }
    
    // Finalize the structure and output results
    void releaseOps(){
        cudaFree(dev_ops);
        cudaFree(dev_ans);
        cudaFree(dev_inputs);
        return;
    }
    
    ~CudaBST(){
        delete [] rest_idx;
        delete [] keys;
        delete [] values;
        delete [] parent;
        delete [] children;
        cudaFree(dev_rest);
        cudaFree(dev_root);
        cudaFree(sync);
        cudaFree(dev_rest_idx);
        cudaFree(dev_keys);
        cudaFree(dev_values);
        cudaFree(dev_parent);
        cudaFree(dev_children);
    }
};

class TestCudaBST{
    int N_ops, *inputs, *cuda_rst, *serial_rst;
    char *ops;
    CudaBST cuda_bst;
    inline int getData(){ return rand()%MAX_DATA; }
public:
    TestCudaBST():cuda_bst(1024, 1024){}
    
    void getOpsList(int n1, int n2, int n_search = 1E6, int n_at = 1E6){
        srand(0);
        vector<int> in,tmp;
        unordered_set<int> tmp_keys;
        string op;
        in.push_back(0);
        in.push_back(0);
        in.push_back(0);
        op += "efz";
        
        // Insert a bunch of data
        assert(n1 < MAX_SIZE-1);
        for(int i=0;i<n1;++i){
            op += "i";
            int u = getData();
            in.push_back(u*MAX_DATA + getData());
            tmp_keys.insert(u);
        }
        in.push_back(0);
        in.push_back(0);
        in.push_back(0);
        op += "efz";
        for(int i=0;i<n_search;++i){
            op += "s";
            in.push_back(rand()%MAX_DATA);
        }
        tmp.assign(tmp_keys.begin(), tmp_keys.end());
        for(int i=0;i<n_at;++i){
            op += "g";
            in.push_back(tmp[rand()%(int)tmp.size()]);
        }
        
        // Remove some of data from the tree
        assert(n2<=(int)tmp_keys.size());
        for(int i=0;i<n2;++i){
            op += "r";
            int u = *tmp_keys.begin();
            tmp_keys.erase(tmp_keys.begin());
            in.push_back(u);
        }
        
        in.push_back(0);
        in.push_back(0);
        in.push_back(0);
        op += "efz";
        for(int i=0;i<n_search;++i){
            op += "s";
            in.push_back(rand()%MAX_DATA);
        }
        tmp.assign(tmp_keys.begin(), tmp_keys.end());
        for(int i=0;i<n_at;++i){
            op += "g";
            in.push_back(tmp[rand()%(int)tmp.size()]);
        }
        
        // Insert some data again
        while((int)tmp_keys.size() < MAX_SIZE-1){
            op += "i";
            int u = getData();
            in.push_back(u*MAX_DATA + getData());
            tmp_keys.insert(u);
        }
        
        in.push_back(0);
        in.push_back(0);
        in.push_back(0);
        op += "efz";
        for(int i=0;i<n_search;++i){
            op += "s";
            in.push_back(rand()%MAX_DATA);
        }
        tmp.assign(tmp_keys.begin(), tmp_keys.end());
        for(int i=0;i<n_at;++i){
            op += "g";
            in.push_back(tmp[rand()%(int)tmp.size()]);
        }
        
        assert((int)op.size() == (int)in.size());
        N_ops = (int)op.size();
        inputs = new int [N_ops];
        ops = new char [N_ops];
        serial_rst = new int [N_ops];
        cuda_rst = new int [N_ops];
        memset(serial_rst, 0, sizeof(serial_rst));
        memset(cuda_rst, 0, sizeof(cuda_rst));
        for(int i=0;i<N_ops;++i) ops[i] = op[i], inputs[i] = in[i];
        cout<<"Randomly Generating "<<N_ops<<" different operations"<<endl<<endl;
    }
    
    void checkSerial(){
        map<int, int> test_map;
        cout<<"=====================================Serial Run (using STL map)====================================="<<endl;
        clock_t start_time = clock(), end_time;
        for(int i=0;i<N_ops;++i){
            if(ops[i] == 'e') serial_rst[i] = test_map.empty();
            else if(ops[i] == 'f') serial_rst[i] = ((int)test_map.size() == MAX_SIZE - 1);
            else if(ops[i] == 'z') serial_rst[i] = (int)test_map.size();
            else if(ops[i] == 's') serial_rst[i] = test_map.count(inputs[i]);
            else if(ops[i] == 'g') serial_rst[i] = test_map[inputs[i]];
            else if(ops[i] == 'i') test_map[inputs[i]/MAX_DATA] = inputs[i]%MAX_DATA;
            else test_map.erase(inputs[i]);
        }
        end_time = clock();
        double dt = double(end_time - start_time)/CLOCKS_PER_SEC;
        cout<<setprecision(6);
        cout<<"     TIME USAGE:      \n";
        cout<<"     "<<dt<<" s      "<<endl<<endl;
        cout<<"===================================================================================================="<<endl<<endl;
    }
    
    void checkParallel(){
        cuda_bst.init();
        cuda_bst.loadOperations(N_ops, ops, inputs);
        cout<<"=================================Parallel Run (using vecterized BST)================================="<<endl;
        clock_t start_time = clock(), end_time;
        cuda_bst.runOperations(cuda_rst);
        end_time = clock();
        double dt = double(end_time - start_time)/CLOCKS_PER_SEC;
        cout<<setprecision(6);
        cout<<"     TIME USAGE:      \n";
        cout<<"     "<<dt<<" s      "<<endl<<endl;
        cout<<"===================================================================================================="<<endl<<endl;
        cuda_bst.releaseOps();
    }
    int countMistakes(string &which_op){
        int cnt = 0;
        for(int i=0;i<N_ops;++i) if(serial_rst[i] != cuda_rst[i]){
            ++cnt;
            which_op += ops[i];
        }
        return cnt;
    }
    
    ~TestCudaBST(){
        delete [] inputs;
        delete [] ops;
        delete [] serial_rst;
        delete [] cuda_rst;
    }
};

int main(){
    TestCudaBST test;
    string ans;
    test.getOpsList(2048, 500, 1000, 1000);
    test.checkSerial();
    test.checkParallel();
    cout<<"     Mistakes made by cuda:       \n"<<"         "<<test.countMistakes(ans)<<endl<<endl;
    return 0;
}

















