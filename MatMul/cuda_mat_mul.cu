#include<bits/stdc++.h>
using namespace std;

__global__ void matMul(int R,int M,int C,int *A, int *B, int *D){
    int i = blockIdx.x, j = threadIdx.x, block_size = blockDim.x;
    assert(i<R && j<C && block_size == C);
    D[i*C + j] = 0;
    for(int k=0;k<M;++k) D[i*C + j] += A[i*M + k] * B[k*C + j];
    return;
}

int main(int argc, char *argv[]){
    srand(0);
	int R = 256, M = 256, C = 256;
    if(argc > 1) R = stoi(argv[1]);
	if(argc > 2) M = stoi(argv[2]);
	if(argc > 3) C = stoi(argv[3]);
    int **A = new int* [R], **B = new int* [M], **D = new int* [R];
    A[0] = new int [R*M];
    for(int i=1;i<R;++i) A[i] = A[i-1] + M;
    for(int i=0;i<R;++i) for(int j=0;j<M;++j) A[i][j] = rand()%10;
    B[0] = new int [M*C];
    for(int i=1;i<M;++i) B[i] = B[i-1] + C;
    for(int i=0;i<M;++i) for(int j=0;j<C;++j) B[i][j] = rand()%10;
    D[0] = new int [R*C];
    for(int i=1;i<R;++i) D[i] = D[i-1] + C;
    
    clock_t start_time,end_time;
    start_time = clock(); //Record the starting time
    int *dA, *dB, *dD;
    cudaMalloc((void **)&dA, R*M*sizeof(int));
    cudaMalloc((void **)&dB, M*C*sizeof(int));
    cudaMalloc((void **)&dD, R*C*sizeof(int));
    
    // Copy data to divice
    cudaMemcpy(dA, A[0], R*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B[0], M*C*sizeof(int), cudaMemcpyHostToDevice);
    
    // Running code on GPUs
    int block_size = C, n_block = R;
    matMul<<<n_block, block_size>>>(R, M, C, dA, dB, dD);
    cudaMemcpy(D[0], dD, R*C*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    end_time = clock(); // Record the ending time
    double dt = double(end_time - start_time)/CLOCKS_PER_SEC;
    cout<<"Time Usage: "<<dt<<"s\nResults:\n";
    int stride = R*C/10;
    for(int i=0;i<R*C;i+=stride) cout<<D[i/C][i%C]<<' ';
    cout<<endl;
    delete [] A[0];
    delete [] B[0];
    delete [] D[0];
    delete [] A;
    delete [] B;
    delete [] D;
    return 0;
}
