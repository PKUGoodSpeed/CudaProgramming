#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/adjacent_difference.h>

#define REP(i,s,n) for(int (i)=s; (i)<(int)(n);(i)++)
#define RIT(it,c) for(__typeof(c.begin()) it = c.begin();it!=c.end();it++)
#define ALL(x) x.begin(), x.end()
#define SZ(x) (int)(x).size()
#define MSET(m,v) memset(m,v,sizeof(m))

using namespace std;

typedef vector<int> vi;
typedef vector<long> vl;
typedef vector<bool> vb;
typedef vector<double> vd;
typedef pair<int,int> ii;
typedef pair<long, long> ll;
typedef unordered_set<int> ui;

class SerialRunTime{
    int N;
public:
    SerialRunTime(int n):N(n){};
    void seqRun(){
        vector<int> Integers(N);
        cout<<"\n==================================\n";
        clock_t t_start = clock();
        for(int i=0;i<N;++i) Integers[i] = i;
        clock_t t_end = clock();
        cout<<"Sequence Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
        for(int i=0;i<10;++i) cout<<Integers[i]<<' ';
        cout<<"\n==================================\n";
    }
    void genRun(){
        int mod = 1E6;
        vector<int> Integers(N);
        cout<<"\n==================================\n";
        clock_t t_start = clock();
        generate(Integers.begin(), Integers.end(), [&mod](){return rand()%mod;});
        clock_t t_end = clock();
        cout<<"Random Number Generation Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
        for(int i=0;i<10;++i) cout<<Integers[i]<<' ';
        cout<<"\n==================================\n";
    }
    void unaryRun(){
        int mod = 1E6;
        vector<int> Integers(N),ans(N);
        for(int i=0;i<N;++i) Integers[i] = i;
        cout<<"\n==================================\n";
        clock_t t_start = clock();
        transform(Integers.begin(), Integers.end(), ans.begin(), [&mod](int x){return 2*x%mod;});
        clock_t t_end = clock();
        cout<<"Unary Operation transformation Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
        for(int i=0;i<10;++i) cout<<ans[i]<<' ';
        cout<<"\n==================================\n";
    }
    void binaryRun(){
        vector<int> A(N),B(N),C(N);
        for(int i=0;i<N;++i) A[i] = i,B[i] = 5;
        cout<<"\n==================================\n";
        clock_t t_start = clock();
        transform(A.begin(), A.end(), B.begin(), C.begin(), modulus<int>());
        clock_t t_end = clock();
        cout<<"Binary Operation transformation Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
        for(int i=0;i<10;++i) cout<<C[i]<<' ';
        cout<<"\n==================================\n";
    }
    
    void diffRun(){
        vector<int> A(N),B(N);
        for(int i=0;i<N;++i) A[i] = i;
        cout<<"\n==================================\n";
        clock_t t_start = clock();
        adjacent_difference(A.begin(), A.end(), B.begin());
        clock_t t_end = clock();
        cout<<"Adjacent difference Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
        for(int i=0;i<10;++i) cout<<B[i]<<' ';
        cout<<"\n==================================\n";
    }
};

class SerialRunTimeTest{
    SerialRunTime se;
public:
    SerialRunTimeTest(int n):se(n){}
    void run(){
        se.seqRun();
        se.genRun();
        se.unaryRun();
        se.binaryRun();
        se.diffRun();
    }
};

class ThrustRunTime{
	int N;
public:
	thrust::host_vector<int> A;
 	thrust::device_vector<int> dA;
 	struct getRand{
	private:
 		thrust::uniform_int_distribution<int> g;
		thrust::minstd_rand rng;
	public:
		getRand(int l, int u):g(l, u+1){}
 		__host__ __device__
 		int operator ()(){ return g(rng);}
 	};
 	struct Dop{
 		int M;
 		Dop(int m):M(m){}
 		__host__ __device__
 		int operator ()(int x){ return (2*x)%M;}
 	};
 	ThrustRunTime(int n):N(n){
		A.resize(N);
		dA.resize(N);
	}
 	void seqRun(){
 		cout<<"\n==================================\n";
 		clock_t t_start = clock();
 		thrust::sequence(dA.begin(), dA.end());
 		clock_t t_end = clock();
 		cout<<"Sequence Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
 		thrust::copy(dA.begin(), dA.end(), A.begin());
 		for(int i=0;i<10;++i) cout<<A[i]<<' ';
 		cout<<"\n==================================\n";
 	}
 	void genRun(){
 		int mod = 1E6;
 		getRand g(0, mod);
		thrust::device_vector<int> dB(N);
		cout<<"\n==================================\n";
 		clock_t t_start = clock();
 		thrust::generate(dB.begin(), dB.end(), g);
 		clock_t t_end = clock();
 		cout<<"Random Number Generation Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
 		thrust::copy(dB.begin(), dB.end(), A.begin());
 		for(int i=0;i<N;i+=N/10) cout<<A[i]<<' ';
 		cout<<"\n==================================\n";
	}
	void unaryRun(){
 		int mod = 1E6;
 		Dop unary(mod);
 		thrust::device_vector<int> dB(N);
 		thrust::sequence(dB.begin(), dB.end());
 		cout<<"\n==================================\n";
 		clock_t t_start = clock();
 		thrust::transform(dB.begin(), dB.end(), dA.begin(), unary);
 		clock_t t_end = clock();
 		cout<<"Unary Operation transformation Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
 		thrust::copy(dA.begin(), dA.end(), A.begin());
 		for(int i=0;i<10;++i) cout<<A[i]<<' ';
 		cout<<"\n==================================\n";
	}
 	void binaryRun(){
 		thrust::device_vector<int> dB(N),dC(N);
 		thrust::sequence(dB.begin(), dB.end());
 		thrust::fill(dC.begin(), dC.end(), 5);
 		cout<<"\n==================================\n";
 		clock_t t_start = clock();
 		thrust::transform(dB.begin(), dB.end(), dC.begin(), dA.begin(), thrust::modulus<int>());
 		clock_t t_end = clock();
 		cout<<"Binary Operation transformation Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
 		thrust::copy(dA.begin(), dA.end(), A.begin());
 		for(int i=0;i<10;++i) cout<<A[i]<<' ';
 		cout<<"\n==================================\n";
 	}
 
 	void diffRun(){
 		thrust::device_vector<int> dB(N);
 		thrust::sequence(dB.begin(), dB.end());
 		cout<<"\n==================================\n";
 		clock_t t_start = clock();
 		thrust::adjacent_difference(dB.begin(), dB.end(), dA.begin());
 		clock_t t_end = clock();
 		cout<<"Adjacent difference Time Usage: "<<double(t_end-t_start)/CLOCKS_PER_SEC<<" s\nCheck Answer:"<<endl;
 		thrust::copy(dA.begin(), dA.end(), A.begin());
 		for(int i=0;i<10;++i) cout<<A[i]<<' ';
 		cout<<"\n==================================\n";
 	}
};
 
class ThrustRunTimeTest{
 	ThrustRunTime se;
public:
 	ThrustRunTimeTest(int n):se(n){}
 	void run(){
 		se.seqRun();
 		se.genRun();
 		se.unaryRun();
	 	se.binaryRun();
 		se.diffRun();
 	}
};


int main(int argc, char *argv[]){
    std::ios_base::sync_with_stdio(false),cin.tie(0),cout.tie(0);
    if(argc <= 1 || argv[1][0] == 'S') {
        SerialRunTimeTest test(1<<28);
        test.run();
    }
    else{
       	ThrustRunTimeTest test(1<<28);
        test.run();
    }
    return 0;
}
