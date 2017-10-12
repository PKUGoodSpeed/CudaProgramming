# cxx
CXX = g++
CXXFLAGS = -std=c++11 -I. -O3 -g -Wall

# cc
CC = gcc
CFLAGS = -g

# nvcc
NVCC      = nvcc
NVCCFLAGS = -std=c++11 -O3

# ld
LDFLAGS = -L/usr/local/cuda/lib64
LDLIBS = -lcudart

# dep
DEPS = inner.hpp

# app

APP = app

# obj
ODIR = objs
_OBJ = inner.o main.o 
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

$(ODIR)/%.o: main.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(APP): $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
