all: app

app: cvg.cu
	/usr/local/cuda-8.0/bin/nvcc -std=c++11 -arch=sm_35 -rdc=true cvg.cu -o app

clean:
	rm app
