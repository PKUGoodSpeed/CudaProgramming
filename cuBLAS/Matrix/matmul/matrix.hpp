#ifndef matrix_h
#define matrix_h

#include<vector>
using namespace std;

struct cpu{};
struct gpu{};
struct cublus{};

template <typename DEVICE_TYPE, typename DATA_TYPE>
class MatrixMultiplication{
public:
	vector<DATA_TYPE> operator ()(const vector<DATA_TYPE> &A, const vector<DATA_TYPE> &B, int rA, int cA, int rB, int cB);
};

#endif
