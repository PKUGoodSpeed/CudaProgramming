all: app

app: gpu_fm.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -O3 gpu_fm.cu -o app

clean:
	rm *app