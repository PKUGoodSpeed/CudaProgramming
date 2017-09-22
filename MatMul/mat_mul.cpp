#include<bits/stdc++.h>
using namespace std;

void matMul(int R,int M,int C,int **A, int **B, int **C){
    for(int i=0;i<R;++i) for(int j=0;j<C;++j){
        C[i][j] = 0;
        for(int k=0;k<M;++k) C[i][j] += A[i][k] * B[k][j];
    }
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
    matMul(R, M, C, A, B, C);
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
