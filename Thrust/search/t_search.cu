#include<bits/stdc++.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/generate.h>
#include<thrust/sort.h>
#include<thrust/copy.h>
#include<thrust/find.h>

using namespace std;

const int mod = 2E7;

class SearchMachine{
	thrust::host_vector<int> A;
	thrust::device_vector<int> dA;
	int N;
public:
	SearchMachine(int n):N(n){
		A.resize(N);
		dA.resize(N);
		thrust::generate(A.begin(), A.end(), [&](){return rand()%mod;});
		thrust::copy(A.begin(), A.end(), dA.begin());
	}
	void processQueries(thrust::host_vector<int> &ans, thrust::host_vector<int> &input){
		cout<<"=============Begin processing queries=============="<<endl<<endl;
		clock_t start_time = clock(), end_time;
		ans.resize(input.size());
		for(int i=0;i<(int)input.size();++i) {
			if(thrust::find(dA.begin(), dA.end(), input[i])==dA.end()) ans[i] = 0;
			else ans[i] = 1;
		}
		end_time = clock();
		cout<<"=============Complete proessing queries============"<<endl<<endl;
		cout<<"Time Usage: "<<double(end_time - start_time)/CLOCKS_PER_SEC;
		cout<<" s"<<endl<<endl<<endl;
		return ;
	}
};


int main(){
	srand(0);
	int N = 1<<24 , M = 1<<8;
	thrust::host_vector<int> Q(M), ans;
	for(int i=0;i<M;++i) Q[i] = rand()%mod;
	SearchMachine s(N);
	s.processQueries(ans, Q);
	for(int i=0;i<M;i+=M/10) cout<<ans[i]<<' ';
	cout<<endl<<endl;
	return 0;
}
