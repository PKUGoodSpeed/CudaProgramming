#ifndef lrg_h
#define lrg_h

#include <vector>
using namespace std;

struct cpu{};
struct gpu{};

template<DEVICE_TYPE, DATA_TYPE>
class LinearRegression{
	DATA_TYPE **input;
	DATA_TYPE *output, *weights;
	int N_samp, N_feat;
public:
	LinearRegression(const vector<vector<DATA_TYPE>> &x_train, const vector<DATA_TYPE> &y_train);
	void operator ()(int N_step, DATA_TYPE learning_rate);
	DATA_TYPE getError(const vector<vector<DATA_TYPE>> &x_test, const vector<DATA_TYPE> &y_test);
	vector<DATA_TYPE> getWeights();
};

#endif
