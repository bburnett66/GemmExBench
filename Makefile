

CUDA_ARCH_FLAGS := -arch=sm_70
CC_FLAGS += --std=c++17 $(CUDA_ARCH_FLAGS)
LIB_FLAGS += -lcublas

EXE = gemmEx 

all: $(EXE)

% : %.cu
	nvcc $< $(CC_FLAGS) $(LIB_FLAGS) -o $@

clean:
	rm -f $(EXE)
