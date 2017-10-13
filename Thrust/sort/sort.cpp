#include <bits/stdc++.h>
using namespace std;

int main(){
	int N = 1<<25, mod = 1E6;
	srand(0);
	vector<int> nums(N);
	clock_t start_time = clock(), end_time;
	/* Generating data testing*/
	generate(nums.begin(), nums.end(), [&](){return rand()%mod;});
	end_time = clock();
	cout<<"=====================Generating Data Time Usage========================"<<endl<<endl;
	cout<<"\t\t"<<double(end_time-start_time)/CLOCKS_PER_SEC<<" s\t\t"<<endl<<endl;
	cout<<"======================================================================="<<endl;
	for(int i=0, block = N/10;i<N; i+=block) cout<<nums[i]<<' ';
	cout<<endl<<endl<<endl;

	/* Sorting testing*/
	start_time = clock();
	sort(nums.begin(), nums.end());
	end_time = clock();
	cout<<"===========================Sorting Time Usage=========================="<<endl<<endl;
	cout<<"\t\t"<<double(end_time-start_time)/CLOCKS_PER_SEC<<" s\t\t"<<endl<<endl;
	cout<<"======================================================================="<<endl;
	for(int i=0, block = N/10;i<N; i+=block) cout<<nums[i]<<' ';
	cout<<endl<<endl<<endl;

	return 0;
}
