# Variables
MODE=Release
NUM_THREADS=16
USE_BP=true
USE_64_BIT=false
ORDER_METHOD=degree_eigen_cent
N_ROOTS=15
SCHEDULE=dynamic,256
SMART_DIST_CACHE_CUTOFF=0
DPSL_BIN_FILE=dpsl
PSL_BIN_FILE=psl
ELIM_MIN=false
ELIM_LEAF=false
COMPRESS=false
COMP_LVL=-1

ifeq ($(COMP_LVL), 0)
$(info Using compression level 0...)
override COMPRESS = false
override ELIM_LEAF = false
override ELIM_MIN = false
else ifeq ($(COMP_LVL), 1)
$(info Using compression level 1...)
override COMPRESS = true
override ELIM_LEAF = false
override ELIM_MIN = false
else ifeq ($(COMP_LVL), 2)
$(info Using compression level 2...)
override COMPRESS = true
override ELIM_LEAF = false
override ELIM_MIN = true
else ifeq ($(COMP_LVL), 3)
$(info Using compression level 3...)
override COMPRESS = true
override ELIM_LEAF = true
override ELIM_MIN = true
endif

CXX_COMPILER=g++
MPICXX_COMPILER=mpic++
CXX_FLAGS= -fopenmp -std=c++17 -DNUM_THREADS=$(NUM_THREADS) -DORDER_METHOD=\"$(ORDER_METHOD)\" -DN_ROOTS=$(N_ROOTS) -DSCHEDULE=$(SCHEDULE) -DSMART_DIST_CACHE_CUTOFF=$(SMART_DIST_CACHE_CUTOFF) -DELIM_MIN=$(ELIM_MIN) -DELIM_LEAF=$(ELIM_LEAF)
CXX_RELEASE_FLAGS= -O3
CXX_DEBUG_FLAGS= -O0 -DDEBUG -g
CXX_PROFILE_FLAGS= -O1 -g -fno-inline -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address
PSL_SOURCE_FILES=src/main_psl.cpp src/psl.cpp src/bp.cpp src/common.cpp
DPSL_SOURCE_FILES=src/main_dpsl.cpp src/dpsl.cpp src/cut.cpp src/psl.cpp src/bp.cpp src/common.cpp
DPSL_FLAGS= -DUSE_GLOBAL_BP=$(USE_BP) -DUSE_LOCAL_BP=false -DGLOBAL_COMPRESS=$(COMPRESS) -DLOCAL_COMPRESS=false -DBIN_DPSL 
PSL_FLAGS= -DUSE_GLOBAL_BP=false -DUSE_LOCAL_BP=$(USE_BP) -DGLOBAL_COMPRESS=false -DLOCAL_COMPRESS=$(COMPRESS) -DBIN_PSL


CUDA_COMPILER=nvcc
CUDA_ARCH=sm_60
CUDA_FLAGS= -std=c++14 -DKERNEL_MODE=$(KERNEL_MODE) -arch=$(CUDA_ARCH) -Xcompiler -fopenmp
CUDA_RELEASE_FLAGS= -O3
CUDA_DEBUG_FLAGS= -O0 -g --generate-line-info
CUDA_SOURCE_FILES=src/gpsl.cu
CUDA_LIB_PATH=$(CUDA_HOME)/lib64
GPSL_FLAGS= -DBIN_GPSL

MODE_LOWER=$(shell echo $(MODE) | tr '[:upper:]' '[:lower:]')
ifeq ($(MODE_LOWER) , debug)
$(info Building in Debug mode...)
CXX_MODE_FLAGS = $(CXX_DEBUG_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_DEBUG_FLAGS)
else ifeq ($(MODE_LOWER) , profile)
$(info Building in Profile mode...)
CXX_MODE_FLAGS = $(CXX_PROFILE_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_PROFILE_FLAGS)
else
$(info Building in Release mode...)
CXX_MODE_FLAGS = $(CXX_RELEASE_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_RELEASE_FLAGS)
endif

ifeq ($(USE_64_BIT) , true)
	CXX_FLAGS := $(CXX_FLAGS) -DUSE_64_BIT
endif

dpsl:
	$(MPICXX_COMPILER) $(DPSL_SOURCE_FILES) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(DPSL_FLAGS) -o $(DPSL_BIN_FILE)
psl:
	$(CXX_COMPILER) $(PSL_SOURCE_FILES) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(PSL_FLAGS) -o $(PSL_BIN_FILE)
gpsl:
	$(CUDA_COMPILER) -c $(CUDA_SOURCE_FILES) $(CUDA_FLAGS) $(CUDA_MODE_FLAGS) $(GPSL_FLAGS) -o gpsl.o
	$(MPICXX_COMPILER) -c $(CXX_SOURCE_FILES) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(GPSL_FLAGS)
	$(MPICXX_COMPILER) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(GPSL_FLAGS) -L$(CUDA_LIB_PATH) *.o -lcudart -o gpsl
	rm *.o

