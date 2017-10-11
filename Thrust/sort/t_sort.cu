#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>


using namespace std;

int main(){
	int N = 1<<25, mod = 1E6;
	srand(0);
	vector<int> testing(N);
	//thrust::host_vector<int> nums(N);
	clock_t start_time = clock(), end_time;
	/* Generating data testing*/
	generate(testing.begin(), testing.end(), [&](){return rand()%mod; });
	for(int i=0;i<N;i+=N/10) cout<<testing[i]<<' ';
	cout<<endl;
	thrust::host_vector<int> nums(testing.begin(), testing.end());
	end_time = clock();
	cout<<"=====================Generating Data Time Usage========================"<<endl<<endl;
	cout<<"\t\t"<<double(end_time-start_time)/CLOCKS_PER_SEC<<" s\t\t"<<endl<<endl;
	cout<<"======================================================================="<<endl;
	for(int i=0, block = N/10;i<N; i+=block) cout<<nums[i]<<' ';
	cout<<endl<<endl<<endl;
	
	/* Sorting testing*/
	start_time = clock();
	thrust::device_vector<int> d_vec = testing;
	thrust::sort(d_vec.begin(), d_vec.end());
	end_time = clock();
	cout<<"===========================Sorting Time Usage=========================="<<endl<<endl;
	cout<<"\t\t"<<double(end_time-start_time)/CLOCKS_PER_SEC<<" s\t\t"<<endl<<endl;
	thrust::copy(d_vec.begin(), d_vec.end(), testing.begin());
	cout<<"======================================================================="<<endl;
	for(int i=0, block = N/10;i<N; i+=block) cout<<testing[i]<<' ';
	cout<<endl<<endl<<endl;

	return 0;
}
