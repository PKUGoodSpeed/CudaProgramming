all: app

app: gpu_fm.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -O3 -lcudart -lcublas -lcublas_device -lcudadevrt gpu_fm.cu -o app
	
pre_app: preproc.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -O3 -lcudart -lcublas -lcublas_device -lcudadevrt preproc.cu -o app

clean:
	rm *app