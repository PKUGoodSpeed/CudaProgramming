all: float

float: float.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -O3 -arch=sm_35 -rdc=true float.cu -o f_app

double: double.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -O3 -arch=sm_35 -rdc=true double.cu -o d_app

clean:
	rm *app
