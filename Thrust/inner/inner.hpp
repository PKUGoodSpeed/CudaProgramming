#ifndef inner_h
#define inner_h

#include <vector>
using namespace std;

int innerSerial(const vector<int> &x, const vector<int> &y);
int innerParallel(const vector<int> &x, const vector<int> &y);

int innerProduct(const vector<int> &x, const vector<int> &y,bool parallel = false);

#endif /*vector inner product*/
