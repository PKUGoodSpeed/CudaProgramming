all: regi share global sync

NVCC = /usr/local/cuda-8.0/bin/nvcc
FLAGS = -std=c++11 -arch=sm_35 -O3 -lcudart -lcublas -lcublas_device -lcudadevrt

regi: regi.cu
	$(NVCC) $(FLAGS) regi.cu -o regi_app

share: share.cu
	$(NVCC) $(FLAGS) share.cu -o share_app
	
global: global.cu
	$(NVCC) $(FLAGS) global.cu -o global_app
	
sync: sync.cu
	$(NVCC) $(FLAGS) sync.cu -o sync_app

clean:
	rm *app