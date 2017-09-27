#include<bits/stdc++.h>
using namespace std;
const int MAX_ARRAY_SIZE = 266;

__global__ void stanSum(int N, int *A, int R){
	int i = blockIdx.x, j = threadIdx.x, block_size = blockDim.x;
	__shared__ int tmp[MAX_ARRAY_SIZE];
	assert(MAX_ARRAY_SIZE >= block_size + 2*R);
	int gidx = i*block_size + j;
	int lidx = R + j;
	tmp[lidx] = gidx<N? A[gidx]:0;
	if(j<R){
		tmp[lidx - R] = gidx<R? 0:A[gidx-R];
		tmp[lidx + block_size] = gidx+block_size<N? A[gidx+block_size]:0;
	}
	__syncthreads();
	for(int j=1;j<=R;++j) A[gidx] += tmp[lidx+j] + tmp[lidx-j];
	return;
}

int main(int argc, char *argv[]){
	int N = 256, R = 10, block_size = 32;
	if(argc > 1) N = stoi(argv[1]);
	if(argc > 2) R = stoi(argv[2]);
	if(argc > 3) block_size = stoi(argv[3]);
	int n_block = (N + block_size - 1)/block_size;
	int *A = new int [N];
	// Initializing A
	// memset(A, -1, N*sizeof(int));
	for(int i=0;i<N;++i) A[i] = i;

	clock_t start_time, end_time;
	// Record the starting time.
	start_time = clock();
	int *dA;
	cudaMalloc((void **)&dA, N*sizeof(int));
	cudaMemcpy(dA, A, N*sizeof(int), cudaMemcpyHostToDevice);
    for(int i=0;i<2;++i){
        stanSum<<<n_block, block_size>>>(N, dA, R);
    }
	cudaMemcpy(A, dA, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(A);
	// Record the ending time.
	end_time = clock();
	double dt = double(end_time - start_time)/CLOCKS_PER_SEC;
	cout<<"Time Usage: "<<dt<<"s\nResults:\n";
	int stride = N/10;
	for(int i=0;i<N;i+=stride) cout<<A[i]<<' ';
	cout<<endl;
	delete [] A;
	return 0;
}
