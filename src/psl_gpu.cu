#include "psl.h"
#include "bp.h"

struct CUDALabelSetNode {
  int* data;
  int size;
  int dist;
};

struct CUDALabelSet {
  CUDALabelSetNode* array;
};

__device__ IDType* d_row_ptr = nullptr;
__device__ IDType* d_col = nullptr;
IDType n = 0;
IDType m = 0;
__device__ CUDALabelSet* d_labels = nullptr;
__device__ BPLabel* d_bp_labels = nullptr;

void setDevice(int device){
   cudaSetDevice(device); 
}

void LoadGPU(CSR& csr){
    n = csr.n;
    m = csr.m;
    size_t size = (n+1) * sizeof(IDType);
    cudaMalloc(&d_row_ptr, size);
    cudaMemcpy(d_row_ptr, csr.row_ptr, size, cudaMemcpyHostToDevice);

    size = m * sizeof(IDType);
    cudaMalloc(&d_col, size);
    cudaMemcpy(d_col, csr.col, size, cudaMemcpyHostToDevice);
    
}

void LoadGPU(BP& bp){
    size_t size = bp.bp_labels.size() * sizeof(BPLabel);
    cudaMalloc(&d_bp_labels, size);
    cudaMemcpy(d_bp_labels, bp.bp_labels.data(), size, cudaMemcpyHostToDevice);
}

void LoadGPU()