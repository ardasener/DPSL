# Options
# Can be Debug or Release, Release mode enables optimizations and turns off some debug code
MODE=Release
# Number of threads used by OpenMP sections
NUM_THREADS=16
# Scheduling strategy for OpenMP sections
SCHEDULE=dynamic,256
# Whether Bit-Parallel labels should be constructed & used
USE_BP=true
# Number of roots used for Bit-Parallel labels
N_ROOTS=15
# [EXPERIMENTAL] Enables 64-bit support
USE_64_BIT=false
# Metric used to rank/order the vertices (see, src/external/order/order.hpp for the supported methods)
ORDER_METHOD=degree
# [EXPERIMENTAL] Vertices with less labels than this will be processed without using the distance cache
SMART_DIST_CACHE_CUTOFF=0
# Sets the output binary name for DPSL
DPSL_BIN_FILE=dpsl
# Sets the output binary name for PSL
PSL_BIN_FILE=psl
# Toggles elimination of local minimum nodes (optimization from PSL*)
ELIM_MIN=false
# Toggles elimination of leaf (degree 1) nodes
ELIM_LEAF=false
# Toggles the compression of the graph by removing identical nodes (optimization from PSL+)
COMPRESS=false
# Override for the compression and elimination options (0 -> no compression, 1 -> COMPRESS, 2 -> COMPRESS + ELIM_MIN, 3 -> COMPRESS + ELIM_MIN + ELIM_LEAF)
COMP_LVL=-1
# Prunes nodes prematurely by keeping the maximum rank of the last label set additions and updating the neighbors before the pull operation
MAX_RANK_PRUNE=true
# Weights used for the partition support uniform, degree, degree_log
PART_WEIGHTS=degree_log
# An attempt at a fix for load balance issues in partitioning due to compression (Setting to -1 turns it off)
PART_LB_OFFSET=100
# Enables mt-kahypar support (Set to "NONE" for off, "DEF" for default, "QUAL" for quality)
MT_KAHYPAR=NONE
# Shuffles upper percentage of the order (Used to demonstrate the importance of order)
ORDER_SHUFFLE=0
# If set to true will call the compiler with ccache
CCACHE=false

# Directory for the dependencies, should contain .a and .h files for pulp and metis
# You can use the scripts/get_deps.sh script to get them automatically on most Linux systems
LIBS_DIR=libs


# Compression level overrides the COMPRESS, ELIM_MIN, ELIM_LEAF options
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

# If BP roots are set to zero, they are turned off
ifeq ($(N_ROOTS), 0)
override USE_BP = false
override N_ROOTS = 15
endif


# C++ flags
CXX_COMPILER=g++
MPICXX_COMPILER=mpic++
CXX_FLAGS= -fopenmp -std=c++17 -DNUM_THREADS=$(NUM_THREADS) -DORDER_METHOD=\"$(ORDER_METHOD)\" -DN_ROOTS=$(N_ROOTS) -DSCHEDULE=$(SCHEDULE) -DSMART_DIST_CACHE_CUTOFF=$(SMART_DIST_CACHE_CUTOFF) -DELIM_MIN=$(ELIM_MIN) -DELIM_LEAF=$(ELIM_LEAF) -DMAX_RANK_PRUNE=$(MAX_RANK_PRUNE) -DPART_WEIGHTS=\"$(PART_WEIGHTS)\" -DPART_LB_OFFSET=$(PART_LB_OFFSET) -DORDER_SHUFFLE=$(ORDER_SHUFFLE)
CXX_RELEASE_FLAGS= -O3
CXX_DEBUG_FLAGS= -O0 -DDEBUG -g
CXX_FAST_DEBUG_FLAGS= -O3 -DDEBUG -g
CXX_PROFILE_FLAGS= -O1 -g -fno-inline -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize=address
PSL_SOURCE_FILES=src/main_psl.cpp src/utils/*.cpp src/psl/*.cpp
DPSL_SOURCE_FILES=src/main_dpsl.cpp src/utils/*.cpp src/psl/*.cpp src/dpsl/*.cpp
DPSL_FLAGS= -DUSE_GLOBAL_BP=$(USE_BP) -DUSE_LOCAL_BP=false -DGLOBAL_COMPRESS=$(COMPRESS) -DLOCAL_COMPRESS=false -DBIN_DPSL -L $(LIBS_DIR) -I $(LIBS_DIR) -Wl,-rpath $(LIBS_DIR) -l:libmetis.so -l:libpulp.a -l:libmtmetis.so -l:libwildriver.so
PSL_FLAGS= -DUSE_GLOBAL_BP=false -DUSE_LOCAL_BP=$(USE_BP) -DGLOBAL_COMPRESS=false -DLOCAL_COMPRESS=$(COMPRESS) -DBIN_PSL

# CUDA flags
CUDA_COMPILER=nvcc
CUDA_ARCH=sm_60
CUDA_FLAGS= -std=c++14 -DKERNEL_MODE=$(KERNEL_MODE) -arch=$(CUDA_ARCH) -Xcompiler -fopenmp
CUDA_RELEASE_FLAGS= -O3
CUDA_DEBUG_FLAGS= -O0 -g --generate-line-info
CUDA_FAST_DEBUG_FLAGS= -O3 -DDEBUG -g
CUDA_SOURCE_FILES=src/gpsl.cu
CUDA_LIB_PATH=$(CUDA_HOME)/lib64
GPSL_FLAGS= -DBIN_GPSL

# Debug/Release mode flags
MODE_LOWER=$(shell echo $(MODE) | tr '[:upper:]' '[:lower:]')
ifeq ($(MODE_LOWER) , debug)
$(info Building in Debug mode...)
CXX_MODE_FLAGS = $(CXX_DEBUG_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_DEBUG_FLAGS)
else ifeq ($(MODE_LOWER), fast_debug)
$(info Building in Fast-Debug mode...)
CXX_MODE_FLAGS = $(CXX_FAST_DEBUG_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_FAST_DEBUG_FLAGS)
else ifeq ($(MODE_LOWER) , profile)
$(info Building in Profile mode...)
CXX_MODE_FLAGS = $(CXX_PROFILE_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_PROFILE_FLAGS)
else
$(info Building in Release mode...)
CXX_MODE_FLAGS = $(CXX_RELEASE_FLAGS)
CUDA_MODE_FLAGS = $(CUDA_RELEASE_FLAGS)
endif

# Experimental 64-bit option
ifeq ($(USE_64_BIT) , true)
$(info Enabling 64-bit support...)
CXX_FLAGS := $(CXX_FLAGS) -DUSE_64_BIT
endif

ifeq ($(MT_KAHYPAR) , QUAL)
$(info Enabling MT-Kahypar Quality...)
DPSL_FLAGS := $(DPSL_FLAGS) -l:libmtkahypargq.so -DENABLE_MT_KAHYPAR
else ifeq ($(MT_KAHYPAR), DEF)
$(info Enabling MT-Kahypar Default...)
DPSL_FLAGS := $(DPSL_FLAGS) -l:libmtkahypargraph.so -DENABLE_MT_KAHYPAR
endif

MPICXX_COMPILER_FINAL=$(MPICXX_COMPILER)
CXX_COMPILER_FINAL=$(CXX_COMPILER)
ifeq ($(CCACHE), true)
MPICXX_COMPILER_FINAL=ccache $(MPICXX_COMPILER)
CXX_COMPILER_FINAL=ccache $(CXX_COMPILER)
endif

# Targets
dpsl:
	$(MPICXX_COMPILER_FINAL) $(DPSL_SOURCE_FILES) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(DPSL_FLAGS) -o $(DPSL_BIN_FILE)
psl:
	$(CXX_COMPILER_FINAL) $(PSL_SOURCE_FILES) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(PSL_FLAGS) -o $(PSL_BIN_FILE)
gpsl:
	$(CUDA_COMPILER) -c $(CUDA_SOURCE_FILES) $(CUDA_FLAGS) $(CUDA_MODE_FLAGS) $(GPSL_FLAGS) -o gpsl.o
	$(MPICXX_COMPILER) -c $(CXX_SOURCE_FILES) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(GPSL_FLAGS)
	$(MPICXX_COMPILER) $(CXX_FLAGS) $(CXX_MODE_FLAGS) $(GPSL_FLAGS) -L$(CUDA_LIB_PATH) *.o -lcudart -o gpsl
	rm *.o

