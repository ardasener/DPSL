MODE=Release
NUM_THREADS=16
USE_BP=true
KERNEL_MODE=0

CXX_COMPILER=mpic++
CXX_FLAGS= -fopenmp -std=c++17 -L./libs -l:libmetis.so -Wl,-rpath ./libs -no-pie -DNUM_THREADS=$(NUM_THREADS)
CXX_RELEASE_FLAGS= -O3
CXX_DEBUG_FLAGS= -O0 -g -DDEBUG
CXX_SOURCE_FILES=src/main.cpp
DPSL_FLAGS= -DUSE_GLOBAL_BP=$(USE_BP) -DUSE_LOCAL_BP=false
PSL_FLAGS= -DUSE_GLOBAL_BP=false -DUSE_LOCAL_BP=$(USE_BP) 

CUDA_COMPILER=nvcc
CUDA_ARCH=sm_60
CUDA_FLAGS= -std=c++14 -DKERNEL_MODE=$(KERNEL_MODE) -arch=$(CUDA_ARCH) -Xcompiler -fopenmp
CUDA_RELEASE_FLAGS= -O3
CUDA_DEBUG_FLAGS= -O0 -g --generate-line-info
CUDA_SOURCE_FILES=src/gpsl.cu

ifeq ($(MODE) , Debug)
$(info Building in Debug mode...)
CXX_MODE_FLAGS = $(CXX_DEBUG_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_DEBUG_FLAGS)
else
$(info Building in Release mode...)
CXX_MODE_FLAGS = $(CXX_RELEASE_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_RELEASE_FLAGS)
endif

dpsl:
	$(CXX_COMPILER) $(CXX_SOURCE_FILES) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(DPSL_FLAGS) -o dpsl
psl:
	$(CXX_COMPILER) $(CXX_SOURCE_FILES) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(PSL_FLAGS) -o psl
gpsl:
	$(CUDA_COMPILER) $(CUDA_SOURCE_FILES) $(CUDA_FLAGS) $(CUDA_MODE_FLAGS) -o gpsl
