#include<bits/stdc++.h>
using namespace std;

const double pi = 3.14159265358979323846264;
const double L = 100.;
const double Diff = 1.;
const int MAX_CELL_SIZE = 20;
// In this method, we use squre cells of threads, but we need to specify the size of the square.

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

__global__ void oneIteration(int N, int cell_size, double *cur, double *old, double delta_x,double delta_t){
    
    int b_size = blockDim.x, b_id = blockIdx.x, t_id = threadIdx.x, in_cell = cell_size - 2;
    int n_cell = (N + in_cell -2)/in_cell;
    
    assert(N > 1);
    assert(cell_size <= MAX_CELL_SIZE);
    assert(b_size == cell_size*cell_size);  // Here we only use square cells
    
    __shared__ double tmp_arr[MAX_CELL_SIZE][MAX_CELL_SIZE];  // Define a buffer in the shared memory of one block of threads
    int dX[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    int dY[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    
    // Compute indices in the cell and in the original matrix
    int local_i = t_id/cell_size, local_j = t_id%cell_size;
    int global_i = (b_id/n_cell)*in_cell + local_i, globel_j = (b_id&n_cell)*in_cell + local_j;
    
    // Copy the old data into the buffer array, then synchronize threads
    if(global_i>=0 && global_i<=N && globel_j>=0 && globel_j <= N){
        tmp_arr[local_i][local_j] = old[global_i*(N+1) + globel_j];
    }
    __syncthreads();
    
    // Compute the updated value and store it into *cur
    if(local_i && local_i<cell_size-1 && local_j && local_j<cell_size-1 && global_i && global_i<N && globel_j && globel_j<N){
        double nn_diff = 0;
        for(int k=0;k<8;++k) nn_diff += tmp_arr[local_i+dX[k]][local_j+dY[k]];
        nn_diff -= 8.*tmp_arr[local_i][local_j];
        cur[global_i*(N+1) + globel_j] = tmp_arr[local_i][local_j] + Diff*delta_t*nn_diff/3./pow(delta_x, 2.);
    }
            
    // Maintain Dirichlet Boundary conditions:
    if(!global_i || !globel_j || global_i == N || globel_j == N) {
        cur[global_i*(N+1) + globel_j] = tmp_arr[local_i*(N+1) + local_j];
    }
    return;
}

class DiffEqnSolver{
    int n_grid, array_size, in_cell, cell_size, n_cell;
    double d_x, **val, *cur, *old;
public:
    DiffEqnSolver(int N, int c_size):n_grid(N), cell_size(c_size){
        assert(cell_size > 2 && cell_size <= MAX_CELL_SIZE);
        d_x = L/n_grid;
        in_cell = cell_size - 2;
        n_cell = (n_grid + in_cell - 2)/cell_size;
        array_size = (n_grid+1)*(n_grid+1);
        val = new double* [n_grid + 1];
        val[0] = new double [array_size];
        for(int i=1;i<=n_grid;++i) val[i] = val[i-1] + n_grid + 1;
        cudaMalloc((void **)&cur, array_size*sizeof(double));
        cudaMalloc((void **)&old, array_size*sizeof(double));
        
        // Setting the boundary conditions
        for(int i=0;i<=n_grid;++i){
            val[0][i] = left(i*d_x);
            val[n_grid][i] = right(i*d_x);
            val[i][0] = bottom(i*d_x);
            val[i][n_grid] = top(i*d_x);
        }
    }
    void init(double init_val){
        for(int i=1;i<n_grid;++i) for(int j=1;j<n_grid;++j) val[i][j] = init_val;
    }
    
    // Compute errors (L2 norm)
    double getError(){
        double sum = 0.;
        for(int i=0;i<=n_grid;++i) for(int j=0;j<=n_grid;++j)
            sum += pow(val[i][j] - analytical(i*d_x, j*d_x),2.);
        return sqrt(sum);
    }
    
    // One step of iteration
    void oneStep( double d_t){
        int EDGE = cell + 2;
        cudaMemcpy(old, val[0], array_size*sizeof(double), cudaMemcpyHostToDevice);
        oneIteration<<<n_cell*n_cell, cell_size*cell_size>>>(n_grid, cell_size, cur, old, d_x, d_t);
        cudaMemcpy(val[0], cur, array_size*sizeof(double), cudaMemcpyDeviceToHost);
    }
    
    double runIterations(int N_step, double d_t){
        for(int t=0;t<N_step;++t) oneStep(d_t);
        return getError();
    }
    
    ~MultiBlocks(){
        delete [] val[0];
        delete [] val;
        cudaFree(cur);
        cudaFree(old);
    }
};

int main(){
    MultiBlocks solver(100, 10);
    solver.init(1.);
    int n_batch = 10, n_step = 1000;
    double dt = 0.5;
    cout<<setprecision(3);
    for(int i=1;i<=n_batch;++i) cout<<"Iteration: "<<i<<"\t error:"<<solver.runIterations(n_step, dt)<<endl;
    return 0;
}
