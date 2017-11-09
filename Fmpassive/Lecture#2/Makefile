all: app

scatter: scatter.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -O3 -arch=sm_35 -rdc=true scatter.cu -o scatter_app

scatter2: scatter2.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -O3 -arch=sm_35 -rdc=true scatter2.cu -o scatter2_app

gather: gather.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -O3 -arch=sm_35 -rdc=true gather.cu -o gather_app

shared: shared.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -O3 -arch=sm_35 -rdc=true shared.cu -o shared_app

clean:
	rm *app
