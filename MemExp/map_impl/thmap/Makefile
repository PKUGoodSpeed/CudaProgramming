all: app

app: gpu_map.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -rdc=true gpu_map.cu -o app

clean:
	rm app
