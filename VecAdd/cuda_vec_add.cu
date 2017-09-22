#include<bits/stdc++.h>
using namespace std;

__global__ void vec_add(int N, int *A, int *B, int *C){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    assert( i<N );
    C[i] = A[i] + B[i];
}

int main(int argc, char *argv[]){
    if(argc <= 1){
        cout<<"Error: No Input Dimension";
        return 0;
    }
    srand(0);
    int N = stoi(argv[1]), block_size = 256;
    int n_block = (N+block_size-1)/block_size;
    int *A = new int [N], B = new int [N], C = new int [N];
    for(int i=0;i<N;++i) A[i] = rand()%50;
    for(int i=0;i<N;++i) B[i] = rand()%50;
    clock_t start_time,end_time;
    // Record the starting time
    start_time = clock();
    int *dA, *dB, *dC;
    cudaMalloc((void **)&dA, N*sizeof(int));
    cudaMalloc((void **)&dB, N*sizeof(int));
    cudaMalloc((void **)&dC, N*sizeof(int));

    // Copy data to divice
    cudaMemcpy(dA, A, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N*sizeof(int), cudaMemcpyHostToDevice);

    // Running code on GPUs
    vec_add<<<n_block, block_size>>>(N, dA, dB, dC);
    cudaMemcpy(C, dC, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Record the ending time
    end_time = clock();
    double dt = double(end_time - start_time)/CLOCKS_PER_SEC;
    cout<<"Time Usage: "<<dt<<"s\nResults:\n";
    int stride = N/10;
    for(int i=0;i<N;i+=stride) cout<<C[i]<<' ';
    cout<<endl;
    delete [] A;
    delete [] B;
    delete [] C;
    return 0;
}
