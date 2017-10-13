#include <iostream>
#include <vector>
#include "./inner.hpp"
using namespace std;

int main(){
	vector<int> A{1, 2, 3, 4, 5}, B{6, 7, 8, 9, 10};
	cout << "CPU version:" << endl;
	cout << InnerProd<cpu, int>()(A, B) << endl << endl;
	//cout << "GPU version:" << endl;
	//cout << InnerProd<gpu, int>()(A, B) << endl << endl;
	return 0;
}
