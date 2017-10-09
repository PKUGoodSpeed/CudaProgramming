#include<bits/stdc++.h>
using namespace std;

const int mod = 2E7;

class SearchMachine{
	vector<int> A;
	int N;
	int searchNumber(int x){
		for(int i=0;i<N;++i) if(x == A[i]) return i;
		return -1;
	}
public:
	SearchMachine(int n):N(n){
		A.resize(N);
		generate(A.begin(), A.end(), [&](){return rand()%mod;});
	}
	void processQueries(vector<int> &ans, vector<int> &input){
		cout<<"=============Begin processing queries=============="<<endl<<endl;
		clock_t start_time = clock(), end_time;
		ans.resize(input.size());
		for(int i=0;i<(int)input.size();++i) {
			if(find(A.begin(), A.end(), input[i])==A.end()) ans[i] = 0;
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
	vector<int> Q(M), ans;
	for(int i=0;i<M;++i) Q[i] = rand()%mod;
	SearchMachine s(N);
	s.processQueries(ans, Q);
	for(int i=0;i<M;i+=M/10) cout<<ans[i]<<' ';
	cout<<endl<<endl;
	return 0;
}
