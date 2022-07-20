#ifndef GPSL_CU
#define GPSL_CU

#include "gpsl.h"
#include "omp.h"

__device__ void lock(int* mutex){
  while(atomicCAS_block(mutex, 0, 1) != 0);
}

__device__ void unlock(int* mutex){
  atomicExch_block(mutex, 0);
}

__device__ void ArrayManager_Get(ArrayManager* a, int*& ptr, int size){
  
  lock(&(a->mutex));
 
  ptr = a->buffer + a->last_index;
  a->last_index += size;

  unlock(&(a->mutex));
}

__device__ void ArrayManager_LoadBuffer(ArrayManager* a, int* new_buffer){
  a->buffer = new_buffer;
  a->last_index = 0;
  a->mutex = 0;
}

__device__ void LabelSet_Insert(LabelSet* s, int * data, int size, int dist){
  s->array[dist].data = data;
  s->array[dist].size = size;
}

__device__ void LabelSet_LoadArray(LabelSet* s, LabelSetNode* new_array){
  s->array = new_array;
}



void GPSL::Init(){
 
  double start_time = omp_get_wtime();

  size_t total_size = 0; 
  size_t size = 0; 

  size = csr.n*sizeof(int);
  total_size += size;
  cudaMalloc((void**) &device_dist0_buffer, size);

  size = sizeof(int)*32*number_of_warps;
  total_size += size;
  cudaMalloc((void**) &device_array_sizes, size);
  cudaMemset(device_array_sizes, 0, size);

  size = sizeof(int)*(csr.n+1);
  total_size += size;
  cudaMalloc((void**) &device_row_ptr, size);  
  cudaMemcpy(device_row_ptr, csr.row_ptr, size, cudaMemcpyHostToDevice);

  size = sizeof(int)*(csr.m);
  total_size += size;
  cudaMalloc((void**) &device_col, size);  
  cudaMemcpy(device_col, csr.col, size, cudaMemcpyHostToDevice);

  size = sizeof(BPLabel)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_bp, size);  
  // cudaMemcpy(device_bp, bp_labels.data(), size, cudaMemcpyHostToDevice);

  size = sizeof(int)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_ranks, size);  
  // cudaMemcpy(device_ranks, ranks.data(), size, cudaMemcpyHostToDevice);

  size = sizeof(char) * csr.n * (number_of_warps);
  total_size += size;
  cudaMalloc((void**) &device_caches, size);  

  size = sizeof(char) * csr.n * (number_of_warps);
  total_size += size;
  cudaMalloc((void**) &device_owners, size);  

  size = sizeof(LabelSet)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_labels, size);  

  size = sizeof(LabelSetNode)*(csr.n)*MAX_DIST;
  total_size += size;
  cudaMalloc((void**) &device_label_nodes, size);

  size = sizeof(int*)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_new_labels, size); 
  cudaMemset(device_new_labels, 0, size);

  size = sizeof(int)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_new_labels_size, size);  
  cudaMemset(device_new_labels_size, 0, size);

  size = sizeof(int);
  total_size += size;
  cudaMalloc((void**) &device_updated_flag, size);  
  cudaMemset(device_updated_flag, 0, size);

  size = sizeof(ArrayManager);
  total_size += size;
  cudaMalloc((void**) &device_array_manager, size);  

  size = sizeof(bool)*csr.n;
  total_size += size;
  cudaMalloc((void**)&device_should_run_prev, sizeof(bool)*csr.n);
  cudaMemset(device_should_run_prev, 0, sizeof(bool));

  size = sizeof(bool)*csr.n;
  total_size += size;
  cudaMalloc((void**)&device_should_run_next, sizeof(bool)*csr.n);
  cudaMemset(device_should_run_next, 0, sizeof(bool));

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  cout << "Total Memory: " << total_mem / (double) (1024 * 1024) << " MB" << endl;
  cout << "Free Memory: " << free_mem / (double) (1024 * 1024) << " MB" << endl;

  size_t alloc_mem = free_mem * 0.8; // Allocating 80% of the remainder
  alloc_mem -= (alloc_mem % 4); // Ensure it is divisible by 4 (ie, size of an integer)
  cout << "Allocating: " << alloc_mem / (double) (1024*1024) << " MB" << endl;

  cudaMalloc((void**) &device_init_buffer, alloc_mem);  

  double end_time = omp_get_wtime();

  cout << "Finished Init: " << end_time - start_time << " seconds" << endl;
} 

#endif
