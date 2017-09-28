#include<bits/stdc++.h>
using namespace std;

const double pi = 3.14159265358979323846264;
const double L = 550;
const double Diff = 1.;
const int MAX_BLOCK_WIDTH = 32;

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
    // Run one step of iteration with multiple blocks of threads
    int b_size = blockDim.x, b_id = blockIdx.x, t_id = threadIdx.x, in_cell = cell_size - 2;
    int n_cell = (N + in_cell -2)/in_cell;
    
    assert(N > 1);
    assert(cell_size <= MAX_BLOCK_WIDTH);
    assert(b_size == cell_size*cell_size);  // Here we only use square cells
    
    __shared__ double tmp_arr[MAX_BLOCK_WIDTH][MAX_BLOCK_WIDTH];  // Define a buffer in the shared memory of one block of threads
    int dX[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    int dY[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    
    // Compute indices in the cell and in the original matrix
    int local_i = t_id/cell_size, local_j = t_id%cell_size;
    int global_i = (b_id/n_cell)*in_cell + local_i, global_j = (b_id%n_cell)*in_cell + local_j;
    
    // Copy the old data into the buffer array, then synchronize threads
    if(global_i>=0 && global_i<=N && global_j>=0 && global_j<=N){
        tmp_arr[local_i][local_j] = old[global_i*(N+1) + global_j];
		// Maintain the Dirichlet Type Boundary Conditions.
		if(!global_i || !global_j || global_i==N || global_j==N) cur[global_i*(N+1)+global_j] = tmp_arr[local_i][local_j];
    }
    __syncthreads();
    
    // Compute the updated value and store it into *cur
    if(local_i && local_i<cell_size-1 && local_j && local_j<cell_size-1 && global_i && global_i<N && global_j && global_j<N){
        double nn_diff = 0.;
        for(int k=0;k<8;++k) nn_diff += tmp_arr[local_i+dX[k]][local_j+dY[k]];
        nn_diff -= 8.*tmp_arr[local_i][local_j];
        cur[global_i*(N+1) + global_j] = tmp_arr[local_i][local_j] + Diff*delta_t*nn_diff/3./pow(delta_x, 2.);
    }
    return;
}

__global__ void iterationWithOneBlock(int N,int N_step, int nx_thread, double *cur, double *tmp, double delta_x, double delta_t){
    // Run multiple steps of iterations with one block of threads
    int t_id = threadIdx.x, b_size = blockDim.x, t_width = (N-2+nx_thread)/nx_thread;
    assert(b_size == nx_thread*nx_thread);

	// Define neighbor vectors:
	int dX[8] = {0, 0, 1, -1, 1, -1, 1, -1};
	int dY[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    int global_i = (t_id/nx_thread)*t_width, globel_j = (t_id%nx_thread)*t_width;
    for(int step=0;step<N_step;++step){
        for(int i=global_i;i<global_i+t_width;++i) for(int j=globel_j;j<globel_j+t_width;++j){
            if(i && i<N && j && j<N){
                double nn_sum = 0., coeff = Diff*delta_t/3./pow(delta_x, 2.);
                for(int k=0;k<8;++k) nn_sum += cur[(i+dX[k])*(N+1) + j+dY[k]];
                tmp[i*(N+1) + j] = (1.-8.*coeff)*cur[i*(N+1) + j] + coeff*nn_sum;
            }
        }
        __syncthreads();
        for(int i=global_i;i<global_i+t_width;++i) for(int j=globel_j;j<globel_j+t_width;++j){
            if(i && i<N && j && j<N) cur[i*(N+1) + j] = tmp[i*(N+1) + j];
        }
        __syncthreads();
    }
    return;
}

__global__ void cudaGetError(int N, int nx_thread, double *analytical, double *cur, double *sum){
	int t_id = threadIdx.x, b_size = blockDim.x, t_width = (N-2+)
	int global_i = ()
}

class DiffEqnSolver{
    int n_grid, array_size, in_cell, cell_size, n_cell;
    int t_width, nx_thread;
    double d_x, **val, *cur, *old;
public:
    DiffEqnSolver(int N):n_grid(N){
        d_x = L/n_grid;
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
    
    // Get grid size and block size
    void setUpGrid(int c_size){
        cell_size = c_size;
        assert(cell_size > 2 && cell_size <= MAX_BLOCK_WIDTH);
        in_cell = cell_size - 2;
        n_cell = (n_grid + in_cell - 2)/in_cell;
    }
    // One step of iteration with multiple blocks
    void oneStep(double d_t){
        cudaMemcpy(old, val[0], array_size*sizeof(double), cudaMemcpyHostToDevice);
        oneIteration<<<n_cell*n_cell, cell_size*cell_size>>>(n_grid, cell_size, cur, old, d_x, d_t);
        cudaMemcpy(val[0], cur, array_size*sizeof(double), cudaMemcpyDeviceToHost);
    }
    // Run multiple iterations with multiple blocks
    double runIterations(int N_step, double d_t){
        for(int t=0;t<N_step;++t) oneStep(d_t);
        return getError();
    }
    
    // Get block size if we use only one block of threads
    void setUpBlock(int nx_t){
        nx_thread = nx_t;
        assert(nx_thread > 0 && nx_thread <= MAX_BLOCK_WIDTH);
        t_width = (n_grid-2+nx_thread)/nx_thread;
    }
    // Run multiple iterations with only one block of threads
    double runWithOneBlock(int N_step, double d_t){
        cudaMemcpy(cur, val[0], array_size*sizeof(double), cudaMemcpyHostToDevice);
        iterationWithOneBlock<<<1, nx_thread*nx_thread>>>(n_grid, N_step, nx_thread, cur, old, d_x, d_t);
        cudaMemcpy(val[0], cur, array_size*sizeof(double), cudaMemcpyDeviceToHost);
        return getError();
    }

	void fileOutPut(string filename){
        FILE *fp = fopen(filename.c_str(), "w");
        if (fp == NULL) {
            fprintf(stderr, "Can't open output file %s!\n", filename.c_str());
            exit(1);
        }
        for(int i=0;i<=n_grid;++i) for(int j=0;j<=n_grid;++j){
            fprintf(fp, "%lf %lf %lf\n", i*d_x, j*d_x, val[i][j]);
        }
        fclose(fp);
    }
    
    ~DiffEqnSolver(){
        delete [] val[0];
        delete [] val;
        cudaFree(cur);
        cudaFree(old);
    }
};

int main(int argc, char *argv[]){
    int block_width = 16;
    if(argc > 1) block_width = stoi(argv[1]);
    int nL = (int)L;
    DiffEqnSolver solver(nL);
    solver.init(1.);
    int n_batch = 21, n_step = 1000;
    double dt = 0.5;
    cout<<setprecision(3);
    cout<<"Start running iterations:"<<endl;
    clock_t start_time = clock(), end_time;
    solver.setUpGrid(block_width);
    for(int i=1;i<=n_batch;++i){
        if(false){
            string filename = "data"+to_string(i/4);
            solver.fileOutPut(filename);
        }
        cout<<"Iteration: "<<i<<"\t error:"<<solver.runIterations(n_step, dt)<<endl;
    }
    //solver.setUpBlock(block_width);
    //for(int i=1;i<=n_batch;++i) cout<<"Iteration: "<<i<<"\t error:"<<solver.runWithOneBlock(n_step, dt)<<endl;
    end_time = clock();
    cout<<"End running iterations!"<<endl<<endl;
    cout<<"Time spent during iterations: "<<double(end_time-start_time)/CLOCKS_PER_SEC<<"s\n\n\n";
	cout<<"================================================================================"<<endl;
    return 0;
}
