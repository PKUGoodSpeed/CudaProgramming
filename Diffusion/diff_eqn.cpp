#include<bits/stdc++.h>
using namespace std;

const double pi = 3.14159265358979323846264;
const double L = 100.;
const double Diff = 1.;

class SerialDiffEqn{
    int n_grid;
    int dX[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    int dY[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    double **val, **old, d_x, d_y;
    inline double left(double y) { return 0; }
    inline double right(double y) { return 0; }
    inline double bottom(double x) { return 0; }
    inline double top(double x){ return sinh(pi)*sin(x*pi/L); }
    inline double analytical(double x,double y){
        return sinh(y*pi/L)*sin(x*pi/L);
    }
public:
    SerialDiffEqn(int N):n_grid(N){
        d_x = d_y = L/n_grid;
        val = new double* [n_grid+1];
        val[0] = new double [(n_grid+1)*(n_grid+1)];
        for(int i=1;i<=n_grid;++i) val[i] = val[i-1] + n_grid + 1;
        // For iteration
        old = new double* [n_grid+1];
        old[0] = new double [(n_grid+1)*(n_grid+1)];
        for(int i=1;i<=n_grid;++i) old[i] = old[i-1] + n_grid + 1;
        
        for(int i=0;i<=n_grid;++i){
            val[i][0] = bottom(i*d_x);
            val[i][n_grid] = top(i*d_x);
            val[0][i] = left(i*d_y);
            val[n_grid][i] = right(i*d_y);
        }
    }
    
    double getError(){
        double sum = 0.;
        for(int i=0;i<=n_grid;++i) for(int j=0;j<=n_grid;++j)
            sum += pow(val[i][j] - analytical(i*d_x, j*d_y),2.);
        return sqrt(sum);
    }
    
    double oneIteration(double d_t){
        memcpy(old[0], val[0], (n_grid + 1)*(n_grid + 1)*sizeof(double));
        for(int i=1;i<n_grid;++i) for(int j=1;j<n_grid;++j){
            double d_val = 0;
            for(int k=0;k<8;++k) d_val += old[i+dX[k]][j+dY[k]] - old[i][j];
            d_val *= d_t*Diff/(3.*d_x*d_x);
            val[i][j] += d_val;
        }
        return getError();
    }
    
    ~SerialDiffEqn(){
        delete [] val[0];
        delete [] val;
        delete [] old[0];
        delete [] old;
    }
};

int main(){
    SerialDiffEqn solver(100);
    int n_step = 10000;
    double dt = 0.5;
    cout<<setprecision(3);
    for(int t=1;t<=n_step;++t) {
        double e = solver.oneIteration(dt);
        if(t%1000 == 0) cout<<"Iteration: "<<t/1000<<' '<<e<<endl;
    }
    return 0;
}
