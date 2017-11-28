NVCC = /usr/local/cuda-8.0/bin/nvcc
FLAGS = -std=c++11 -arch=sm_35 -O3 -lcudart -lcublas -lcublas_device -lcudadevrt
BIN = apps

all: mkfolder GOrder list OrderList GOrderList GOrderList2 GOrderHandler GLevelOrderList
	
GOrder: mkfolder GOrder_test.cu
	$(NVCC) $(FLAGS) GOrder_test.cu -o $(BIN)/GOrder
	
list: mkfolder list_test.cu
	$(NVCC) $(FLAGS) list_test.cu -o $(BIN)/list
	
OrderList: mkfolder OrderList_test.cu
	$(NVCC) $(FLAGS) OrderList_test.cu -o $(BIN)/OrderList
	
GOrderList: mkfolder GOrderList_test.cu
	$(NVCC) $(FLAGS) GOrderList_test.cu -o $(BIN)/GOrderList
	
GOrderList2: mkfolder GOrderList_test2.cu
	$(NVCC) $(FLAGS) GOrderList_test2.cu -o $(BIN)/GOrderList2

GOrderHandler: mkfolder GOrderHandler_test.cu
	$(NVCC) $(FLAGS) GOrderHandler_test.cu -o $(BIN)/GOrderHandler

GLevelOrderList: mkfolder GLevelOrderList_test.cu
	$(NVCC) $(FLAGS) GLevelOrderList_test.cu -o $(BIN)/GLevelOrderList

mkfolder:
	if [ ! -d "./$(BIN)" ];then     \
		mkdir $(BIN);           \
	fi
	
clean:
	rm -rf ./apps