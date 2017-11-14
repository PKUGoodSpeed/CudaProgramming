all: single global multi

NVCC = /usr/local/cuda-8.0/bin/nvcc
FLAGS = -std=c++11 -arch=sm_35 -O3 -lcudart -lcublas -lcublas_device -lcudadevrt

single: single.cu
	$(NVCC) $(FLAGS) single.cu -o single_app

global: global.cu
	$(NVCC) $(FLAGS) global.cu -o global_app
	
multi: multi.cu
	$(NVCC) $(FLAGS) multi.cu -o multi_app

clean:
	rm *app