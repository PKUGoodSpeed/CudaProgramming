#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#define cu_dot(x, y) thrust::inner_product((x).begin(), (x).end(), (y).begin(), 0)

using namespace std;

typedef thrust::device_vector<int> dvi;

struct cpu{
	const bool nvcc = false;
};

struct gpu{
	const bool nvcc = true;
};

template<typename DEVICE_TYPE, typename DATA_TYPE>
class InnerProd{
public:
	DATA_TYPE operator ()(const vector<DATA_TYPE> &x, const vector<DATA_TYPE> &y);
};

template<>
int InnerProd<cpu, int>:: operator()(const vector<int> &x, const vector<int> &y){
	return inner_product(x.begin(), x.end(), y.begin(), 0);
}


template<>
int InnerProd<gpu, int>:: operator()(const vector<int> &x, const vector<int> &y){
	dvi dx = x, dy = y;
	return cu_dot(dx, dy);
}

int main(){
	vector<int> A = {1, 2, 3, 4, 5,}, B = {6, 7, 8, 9, 10};
	cout<<"CPU version:"<<endl;
	cout<<InnerProd<cpu, int>()(A, B)<<endl<<endl;
	cout<<"GPU version:"<<endl;
	cout<<InnerProd<gpu, int>()(A, B)<<endl<<endl;
	return 0;
}
