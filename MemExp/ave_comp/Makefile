all: float

float: float.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -rdc=true float.cu -o f_app

double: double.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -rdc=true double.cu -o d_app

multi: multi.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -rdc=true multi.cu -o m_app
	
perf: perf.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -rdc=true perf.cu -o p_app

clean:
	rm *app
