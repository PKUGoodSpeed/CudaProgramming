#include<bits/stdc++.h>
using namespace std;

class TryClass{
public:
    __global__ void vec_add(int N, int *A, int *B, int *C){
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if(i < N) C[0] = A[i] * B[i];
    }
};

int main(int argc, char *argv[]){
    int N = 10, block_size = 16;
    srand(0);
    if(argc > 1) N = stoi(argv[1]);
    int n_block = (N+block_size-1)/block_size;
    int *A = new int [N], *B = new int [N], *C = new int [1];
    for(int i=0;i<N;++i) A[i] = 1;
    for(int i=0;i<N;++i) B[i] = 1;
   
    int *dA, *dB, *dC;
    cudaMalloc((void **)&dA, N*sizeof(int));
    cudaMalloc((void **)&dB, N*sizeof(int));
    :cudaMalloc((void **)&dC, N*sizeof(int));

    // Copy data to divice
    cudaMemcpy(dA, A, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N*sizeof(int), cudaMemcpyHostToDevice);
    mid_time1 = clock();

    // Running code on GPUs
    TryClass().vec_add<<<n_block, block_size>>>(N, dA, dB, dC);
    mid_time2 = clock();
    cudaMemcpy(C, dC, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Record the ending time
    end_time = clock();
    double dt = double(end_time - start_time)/CLOCKS_PER_SEC;
    double dt_trans = double(mid_time1 + end_time - start_time - mid_time2)/CLOCKS_PER_SEC;
    cout<<"Data Transfer Time Usage: "<<dt_trans<<"s"<<endl;
    cout<<"Total Time Usage: "<<dt<<"s\nResults:\n";
    int stride = N/10;
    for(int i=0;i<N;i+=stride) cout<<C[i]<<' ';
    cout<<endl;
    delete [] A;
    delete [] B;
    delete [] C;
    return 0;
}
