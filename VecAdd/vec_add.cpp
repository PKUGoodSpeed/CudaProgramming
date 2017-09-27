#include<bits/stdc++.h>
using namespace std;

void vecAdd(int N, int *A, int *B, int *C){
    for(int i=0;i<N;++i) C[i] = A[i]+B[i];
}

int main(int argc, char *argv[]){
    srand(0);
    int N = 10000;
	if(argc>1) N = stoi(argv[1]);
    int *A = new int [N], *B = new int [N], *C = new int [N];
    for(int i=0;i<N;++i) A[i] = rand()%50;
    for(int i=0;i<N;++i) B[i] = rand()%50;
    
    clock_t start_time, end_time;
    // Record the starting time
    start_time = clock();
    vecAdd(N, A, B, C);
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
