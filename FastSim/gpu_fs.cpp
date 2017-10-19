#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <cassert>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cublas_v2.h>
#include "./matrix.hpp"

using namespace std;
typedef thrust::device_vector<float> dvf;


