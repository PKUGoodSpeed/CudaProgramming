#include <numeric>
#include "./inner.hpp"
using namespace std;

template<>
int InnerProd<cpu, int>::operator ()(const vector<int>&x, const vector<int> &y){
	return inner_product(x.begin(), x.end(), y.begin(), 0);
}
