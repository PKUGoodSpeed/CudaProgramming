NVCC = /usr/local/cuda-8.0/bin/nvcc
FLAGS = -std=c++11 -arch=sm_35 -O3 -lcudart -lcublas -lcublas_device -lcudadevrt
BIN = apps

all: mkfolder sim
	
sim: sim.cu
	$(NVCC) $(FLAGS) sim.cu -o $(BIN)/sim
	
mkfolder:
	if [ ! -d "./$(BIN)" ];then     \
		mkdir $(BIN);           \
	fi
	
clean:
	rm -rf ./apps