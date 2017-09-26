#include<bits/stdc++.h>
using namespace std;

void stanSum(int N, int *A, int R){
	int *tmp = new int [N + 2*R];
	memset(tmp, 0, (N + 2*R)*sizeof(int));
	memcpy(tmp + R, A, N*sizeof(int));
	for(int i=0;i<N;++i){
		for(int j=1;j<=R;++j) A[i] += tmp[i+R-j] + tmp[i+R+j];
	}
	delete [] tmp;
	return;
}

int main(int argc, char *argv[]){
	if(argc < 3){
		cout<<"Error: Not Enough Input!"<<endl;
		return 0;
	}
	int N = stoi(argv[1]), R = stoi(argv[2]);
	int *A = new int [N];
	// Initializing A
	// memset(A, -1, N*sizeof(int));
	for(int i=0;i<N;++i) A[i] = i;

	clock_t start_time, end_time;
	// Record the starting time.
	start_time = clock();
    for(int i=0;i<100;++i){
        stanSum(N, A, R);
    }
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
