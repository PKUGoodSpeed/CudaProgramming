#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/random/uniform_int_distribution.h>


using namespace std;
#define t_seq(x) thrust::sequence((x).begin(), (x).end())
#define t_gen(x, g) thrust::generate((x).begin(), (x).end(), (g))
#define t_copy(x, y) thrust::copy((x).begin(), (x).end(), (y).begin())
#define t_tran_u(x, z, u) thrust::transform((x).begin(), (x).end(), (z).begin(), u)
#define t_tran_b(x, y, z, b) thrust::transform((x).begin(), (x).end(), (y).begin(), (z).begin(), b)
#define t_cnt(l ,r) thrust::make_counting_iterator(l), thrust::make_counting_iterator(r)
#define t_sum(x) thrust::reduce((x).begin(), (x).end(), 0)

typedef thrust::host_vector<int> hvi;
typedef thrust::device_vector<int> dvi;
//typedef thrust::default_random_engine t_drng;
//typedef thurst::uniform_int_distribution<int> t_udis;

const int mod = 1E6;

struct f{
	__device__ int operator()(int x){return x*x + x + 1;}
};

template<int lower, int upper>
class gr{
public:
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> g;
	gr():g(lower, upper){}
	__host__ __device__ int operator ()(int idx){
		rng.discard(idx);
		return g(rng);
	}
};

__global__ void fg(dvi &tmp){
	t_tran_u(tmp, tmp, f());
	return;
}

int main(){
	int N = 1<<4;
	hvi h_vec(N);
	dvi d_vec(N);
	t_seq(h_vec);
	for(auto k:h_vec) cout<<k<<' ';
	cout<<endl;
	t_seq(d_vec);
	t_tran_u(d_vec, d_vec, f());
	t_copy(d_vec, h_vec);
	for(auto k:h_vec) cout<<k<<' ';
	cout<<endl;
	thrust::transform(t_cnt(0, N), d_vec.begin(), gr<1,mod>());
	t_copy(d_vec, h_vec);
	for(auto k:h_vec) cout<<k<<' ';
	cout<<endl;
	cout<<gr<1,10>()(2)<<endl;
	cout<<endl<<endl<<endl<<endl;
	fg<<<4, 4>>>(d_vec);
	t_copy(d_vec, h_vec);
	for(auto k:h_vec) cout<<k<<' ';
	cout<<endl;
	return 0;
}
