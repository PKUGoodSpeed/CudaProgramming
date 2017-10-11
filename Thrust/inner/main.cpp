#include "inner.hpp"
#include <bits/stdc++.h>

using namespace std;

int main(){
	int N = 1<<28;
	cout<<"N = "<<N<<endl<<endl;
	vector<int> A(N, 1), B(N ,1);
	clock_t t1 = clock(), t2;
	cout<<innerProduct(A, B, false)<<endl;
	t2 = clock();
	cout<<"Running time for the serial inner product is "<<double(t2 - t1)/CLOCKS_PER_SEC<<" s\n\n";
	cout<<innerProduct(A, B, true)<<endl;
	t1 = clock();
	cout<<"Running time for the parallel inner product is "<<double(t1 - t2)/CLOCKS_PER_SEC<<" s\n\n";
	return 0;
}
