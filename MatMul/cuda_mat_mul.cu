#include<bits/stdc++.h>
using namespace std;

__global__ void matMul(int R,int M,int C,int *A, int *B, int *C){
    int i = blockIdx.x, j = threadIdx.x, block_size = blockDim.x;
    assert(i<R && j<C && block_size == C);
    C[i*C + j] = 0;
    for(int k=0;k<M;++k) C[i*C + j] = A[i*M + k] * B[k*C + j];
    return;
}

int main(int argc, char *argv[]){
    if(argc < 4){
        cout<<"Error: Not Enough Input Dimensions!"<<endl;
        return 0;
    }
    srand(0);
    int R = stoi(argv[1]), M = stoi(argv[2]), C = stoi(argv[3]);
    int **A = new int* [R], **B = new int* [M], **C = new int* [R];
    A[0] = new int [R*M];
    for(int i=1;i<R;++i) A[i] = A[i-1] + M;
    for(int i=0;i<R;++i) for(int j=0;j<M;++j) A[i][j] = rand()%10;
    B[0] = new int [M*C];
    for(int i=1;i<M;++i) B[i] = B[i-1] + C;
    for(int i=0;i<M;++i) for(int j=0;j<C;++j) B[i][j] = rand()%10;
    C[0] = new int [R*C];
    for(int i=1;i<R;++i) C[i] = C[i-1] + C;
    
    clock_t start_time,end_time;
    start_time = clock(); //Record the starting time
    int *dA, *dB, *dC;
    cudaMalloc((void **)&dA, R*M*sizeof(int));
    cudaMalloc((void **)&dB, M*C*sizeof(int));
    cudaMalloc((void **)&dC, R*C*sizeof(int));
    
    // Copy data to divice
    cudaMemcpy(dA, A[0], R*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B[0], M*C*sizeof(int), cudaMemcpyHostToDevice);
    
    // Running code on GPUs
    int block_size = C, n_block = R;
    matMul<<<n_block, block_size>>>(R, M, C, dA, dB, dC);
    cudaMemcpy(C[0], dC, R*C*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    end_time = clock(); // Record the ending time
    double dt = double(end_time - start_time)/CLOCKS_PER_SEC;
    cout<<"Time Usage: "<<dt<<"s\nResults:\n";
    int stride = R*C/10;
    for(int i=0;i<R*C;i+=stride) cout<<C[i/C][i%C]<<' ';
    cout<<endl;
    delete [] A[0];
    delete [] B[0];
    delete [] C[0];
    delete [] A;
    delete [] B;
    delete [] C;
    return 0;
}
