NVCC = /usr/local/cuda-8.0/bin/nvcc
FLAGS = -std=c++11 -arch=sm_35 -O3 -lcudart -lcublas -lcublas_device -lcudadevrt
BIN = apps

all: mkfolder glist gstack gqueue gdeque lglist lgstack lgqueue

glist: mkfolder glist_test.cu
	$(NVCC) $(FLAGS) glist_test.cu -o $(BIN)/glist
	
gstack: mkfolder gstack_test.cu
	$(NVCC) $(FLAGS) gstack_test.cu -o $(BIN)/gstack
	
gqueue: mkfolder gqueue_test.cu
	$(NVCC) $(FLAGS) gqueue_test.cu -o $(BIN)/gqueue

gdeque: mkfolder gdeque_test.cu
	$(NVCC) $(FLAGS) gdeque_test.cu -o $(BIN)/gdeque

lglist: mkfolder lglist_test.cu
	$(NVCC) $(FLAGS) lglist_test.cu -o $(BIN)/lglist
	
lgstack: mkfolder lgstack_test.cu
	$(NVCC) $(FLAGS) lgstack_test.cu -o $(BIN)/lgstack

lgqueue: mkfolder lgqueue_test.cu
	$(NVCC) $(FLAGS) lgqueue_test.cu -o $(BIN)/lgqueue

mkfolder:
	if [ ! -d "./$(BIN)" ];then     \
		mkdir $(BIN);           \
	fi
	
clean:
	rm -rf ./apps