#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>


using namespace std;
const int mod = 1E6;
int main(){
	int N = 1<<25;
	srand(0);
	thrust::host_vector<int> nums(N);
	thrust::device_vector<int> d_vec(N);
	clock_t start_time = clock(), end_time;
	/* Generating data testing*/
	thrust::generate(d_vec.begin(),d_vec.end(), [&](){return rand()%mod;});
	end_time = clock();
	cout<<"=====================Generating Data Time Usage========================"<<endl<<endl;
	cout<<"\t\t"<<double(end_time-start_time)/CLOCKS_PER_SEC<<" s\t\t"<<endl<<endl;
	cout<<"======================================================================="<<endl;
	thrust::copy(d_vec.begin(), d_vec.end(), nums.begin());
	for(int i=0, block = N/10;i<N; i+=block) cout<<nums[i]<<' ';
	cout<<endl<<endl<<endl;
	
	/* Sorting testing*/
	start_time = clock();
	thrust::sort(d_vec.begin(), d_vec.end());
	end_time = clock();
	cout<<"===========================Sorting Time Usage=========================="<<endl<<endl;
	cout<<"\t\t"<<double(end_time-start_time)/CLOCKS_PER_SEC<<" s\t\t"<<endl<<endl;
	thrust::copy(d_vec.begin(), d_vec.end(), nums.begin());
	cout<<"======================================================================="<<endl;
	for(int i=0, block = N/10;i<N; i+=block) cout<<nums[i]<<' ';
	cout<<endl<<endl<<endl;

	return 0;
}
