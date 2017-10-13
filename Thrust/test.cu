#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <bits/stdc++.h>
using namespace std;

int main(){
	thrust::device_vector<int> d_B(5), d_A(5,15);
	cout<<"begin\n\n";
	for(auto b:d_B) cout<<b<<' ';
	cout<<endl;
	for(auto a:d_A) cout<<a<<' ';
	cout<<endl;
	thrust::sequence(d_B.begin(), d_B.end());
	cout<< thrust::inner_product(d_A.begin(), d_A.end(), d_B.begin(), 0)<<endl;
}
