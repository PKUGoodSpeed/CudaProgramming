#ifndef inner_h
#define inner_h

#include<vector>
using namespace std;

struct cpu{};
struct gpu{};

template <typename DEVICE_TYPE, typename DATA_TYPE>
class InnerProd{
public:
	DATA_TYPE operator ()(const vector<DATA_TYPE> &x, const vector<DATA_TYPE> &y);
};

#endif
