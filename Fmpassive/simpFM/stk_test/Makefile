all: app

NVCC = /usr/local/cuda-8.0/bin/nvcc
FLAGS = -std=c++11 -arch=sm_35 -O3 -lcudart -lcublas -lcublas_device -lcudadevrt

app: test.cu
	$(NVCC) $(FLAGS) test.cu -o app
	
clean:
	rm *app