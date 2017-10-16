# cxx
CXX = g++
CXXFLAGS = -I. -std=c++11 -O3 -g -Wall

# cc
CC = gcc
CFLAGS = -g

# nvcc
NVCC = nvcc
NVCCFLAGS = -I. -std=c++11 -O3 -arch=sm_35

# ld and libs
LDFLAGS = 
LIBS = 
CULDFLAGS = -L/usr/local/cuda/lib64/
CULIBS = -lcudart

# dependence
DEPS = inner.hpp

# objs
ODIR = objs
_OBJS = cpu_inner.o main.o
OBJS = $(patsubst %, $(ODIR)/%, $(_OBJS))
CUODIR = cu_objs
_CUOBJS = gpu_inner.o
CUOBJS = $(patsubst %, $(CUODIR)/%, $(_CUOBJS))

all: gpu

cpu: app_cpu

gpu: app_gpu

$(CUODIR)/%.o: %.cu $(CUODIR)
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

$(CUODIR):
	mkdir $(CUODIR)

$(ODIR)/%.o: %.cpp $(ODIR) $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(ODIR):
	mkdir $(ODIR)

app_cpu: $(OBJS)
	$(CXX) -o app_cpu $^ $(CXXFLAGS) $(LDFLAGS) $(LIBS)

app_gpu: $(OBJS) $(CUOBJS)
	$(CXX) -o app_gpu $^ $(CXXFLAGS) $(CULDFLAGS) $(CULIBS)

.PHONY: clean
clean:
	rm -rf $(ODIR) $(CUODIR)
