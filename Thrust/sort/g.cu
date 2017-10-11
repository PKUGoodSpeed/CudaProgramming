#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/random/uniform_int_distribution.h>
using namespace std;

class Rand{
	//const int mod = 1E6;
	thrust::uniform_int_distribution<int> g;
	thrust::default_random_engine rng;
public:
	Rand():g(0, 1000001){}
	__host__ __device__ int operator ()(int idx){
		rng.discard(idx);
		return g(rng);
	}
};

int main(){
	int N = 1<<25;
	srand(0);
	
	thrust::host_vector<int> nums(N);
	thrust::device_vector<int> d_vec(N);
	Rand gr;
	clock_t start_time = clock(), end_time;
	/* Generating data testing*/
	thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(N),
	d_vec.begin(), gr);
	end_time = clock();
	cout<<"=====================Generating Data Time Usage========================"<<endl<<endl;
	cout<<"\t\t"<<double(end_time-start_time)/CLOCKS_PER_SEC<<" s\t\t"<<endl<<endl;
	cout<<"======================================================================="<<endl;
	thrust::copy(d_vec.begin(), d_vec.end(), nums.begin());
	for(int i=0;i<10; ++i) cout<<nums[i]<<' ';
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
