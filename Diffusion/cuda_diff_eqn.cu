#include<bits/stdc++.h>
using namespace std;

const double pi = 3.14159265358979323846264;
const double L = 100.;
const double Diff = 1.;
const int MAX_BLOCK_SIZE = 1048;

/*
  |   coordinate system:
 -|---------------y
  |   x = i * d_x
  |   y = i * d_x
  |
  x
 */

inline double left(double y) { return 0; }
inline double right(double y) { return 0; }
inline double bottom(double x) { return 0; }
inline double top(double x){ return sinh(pi)*sin(x*pi/L); }
inline double analytical(double x,double y){
    return sinh(y*pi/L)*sin(x*pi/L);
}

__global__ void oneIteration(int N, double *cur, double *old, double delta_x,double delta_t){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int dX[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    int dY[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    assert(N > 1);
    if(idx < (N-1)*(N-1)){
        int i = idx/(N-1) + 1, j = idx%(N-1) + 1;
        double d_val = 0;
        for(int k=0;k<8;++k) d_val += old[(i+dX[k])*(N+1) + j+dY[k]];
        cur[i*(N+1) + j] += Diff*delta_t*(d_val - 8.*old[i*(N+1) + j])/3./pow(delta_x, 2.);
    }
    return;
}

class DiffEqnSolver{
    int n_grid, array_size, block_size, n_block;
    double d_x, **val, *cur, *old;
public:
    DiffEqnSolver(int N,int b_size):n_grid(N), block_size(b_size){
        assert(block_size > 0 && block_size <= MAX_BLOCK_SIZE);
        d_x = L/n_grid;
        n_block = ((n_grid-1)*(n_grid-1) + block_size - 1)/block_size;
        array_size = (n_grid+1)*(n_grid+1);
        val = new double* [n_grid + 1];
        val[0] = new double [array_size];
        for(int i=1;i<=n_grid;++i) val[i] = val[i-1] + n_grid + 1;
        cudaMalloc((void **)&cur, array_size*sizeof(double));
        cudaMalloc((void **)&old, array_size*sizeof(double));
        for(int i=0;i<=n_grid;++i){
            val[0][i] = left(i*d_x);
            val[n_grid][i] = right(i*d_x);
            val[i][0] = bottom(i*d_x);
            val[i][n_grid] = top(i*d_x);
        }
    }
    void init(double init_val){
        // Initialize the grid
        for(int i=1;i<n_grid;++i) for(int j=1;j<n_grid;++j) val[i][j] = init_val;
    }
    
    double getError(){
        // Using L2 norm
        double sum = 0.;
        for(int i=0;i<=n_grid;++i) for(int j=0;j<=n_grid;++j)
            sum += pow(val[i][j] - analytical(i*d_x, j*d_x),2.);
        return sqrt(sum);
    }
    
    void oneStep( double d_t){
        cudaMemcpy(old, val[0], array_size*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cur, val[0], array_size*sizeof(double), cudaMemcpyHostToDevice);
        oneIteration<<<n_block, block_size>>>(n_grid, cur, old, d_x, d_t);
        cudaMemcpy(val[0], cur, array_size*sizeof(double), cudaMemcpyDeviceToHost);
    }
    
    double runIterations(int N_step, double d_t){
        for(int t=0;t<N_step;++t) oneStep(d_t);
        return getError();
    }
    
    ~DiffEqnSolver(){
        delete [] val[0];
        delete [] val;
        cudaFree(cur);
        cudaFree(old);
    }
};

int main(int argc, char *argv[]){
    int block_size = 33;
    if(argc > 1) block_size = stoi(argv[1]);
    DiffEqnSolver solver(100, block_size);
    solver.init(1.);
    int n_batch = 20, n_step = 1000;
    double dt = 0.5;
    cout<<setprecision(3);
    cout<<"Start running iterations:"<<endl;
    clock_t start_time = clock(), end_time;
    for(int i=1;i<=n_batch;++i) cout<<"Iteration: "<<i<<"\t error:"<<solver.runIterations(n_step, dt)<<endl;
    end_time = clock();
    cout<<"End running iterations!"<<endl<<endl;
    cout<<"Time spent during iterations: "<<double(end_time-start_time)/CLOCKS_PER_SEC<<"s\n\n\n";
    cout<<"================================================================================"<<endl;
    return 0;
}
