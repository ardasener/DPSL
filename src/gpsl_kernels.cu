#ifndef GPSL_KERNELS_CU
#define GPSL_KERNELS_CU

#include "gpsl_kernels.h"
#include "common.h"
#include "bp.h"
#include "external/order/order.hpp"


// IN GBs
#define ARRAY_MANAGER_ALLOC_SIZE 1
#define GBS_TO_BYTES (1024ULL*1024ULL*1024ULL);


/* #define DEBUG */

/*
  0 -> parallelize at w only
  1 -> parallelize at v only
  2 -> 8 groups for v, 4 threads for w
  3 -> 4 groups for v, 8 threads for w
  4 -> hybrid of 0 and 1
  5 -> hybrid of 2 and 3
  6 -> hybrid of 0,1,2 and 3
  7 -> parallel prune (32 threads at x)
*/
#ifndef KERNEL_MODE
#define KERNEL_MODE 0
#endif

#define SORT_CUT_OFF 32

__device__ void lock(int* mutex){
  while(atomicCAS_block(mutex, 0, 1) != 0);
}

__device__ void unlock(int* mutex){
  atomicExch_block(mutex, 0);
}

__device__ void ArrayManager::Get(int*& ptr, int size){
  
  lock(&mutex);
 
  ptr = buffer + last_index;
  last_index += size;

  unlock(&mutex);
}

__device__ void ArrayManager::LoadBuffer(int* new_buffer){
  buffer = new_buffer;
  last_index = 0;
  mutex = 0;
}

__device__ void LabelSet::Insert(int * data, int size, int dist){
  array[dist].data = data;
  array[dist].size = size;
}

__device__ void LabelSet::LoadArray(LabelSetNode* new_array){
  array = new_array;
}

__device__ bool GPSL_PruneByBp(int u, int v, int d, BPLabel* bp_labels){

  BPLabel &idx_u = bp_labels[u], &idx_v = bp_labels[v];

  for (int i = 0; i < N_ROOTS; ++i) {

      int td = idx_u.bp_dists[i] + idx_v.bp_dists[i];
      if (td - 2 <= d)
	td += (idx_u.bp_sets[i][0] & idx_v.bp_sets[i][0]) ? -2
	      : ((idx_u.bp_sets[i][0] & idx_v.bp_sets[i][1]) |
				 (idx_u.bp_sets[i][1] & idx_v.bp_sets[i][0]))
		  ? -1
		  : 0;
  
      if (td <= d){
	      return true;
      }
    }
    return false;

}


__device__ int GPSL_QueryByBp(int u, int v, BPLabel* bp_labels){

  BPLabel &idx_u = bp_labels[u], &idx_v = bp_labels[v];

  int d = MAX_DIST;
  for (int i = 0; i < N_ROOTS; ++i) {

      int td = idx_u.bp_dists[i] + idx_v.bp_dists[i];
      if (td - 2 <= d)
	td += (idx_u.bp_sets[i][0] & idx_v.bp_sets[i][0]) ? -2
	      : ((idx_u.bp_sets[i][0] & idx_v.bp_sets[i][1]) |
				 (idx_u.bp_sets[i][1] & idx_v.bp_sets[i][0]))
		  ? -1
		  : 0;
  
      if (td < d){
	d = td;
      }
    }

    return d;

}

__device__ bool GPSL_Prune(int u, int v, int d, char* cache, LabelSet* device_labels){
  LabelSet& labels_v = device_labels[v];

  for(int i=0; i<d; i++){
    LabelSetNode& node = labels_v.array[i];

    for(int j=0; j<node.size; j++){
      int w = node.data[j];
      int cache_dist = cache[w];

      if((i + cache_dist) <= d){
	return true;
      }
    }
  }

  return false;

}

__device__ void GPSL_Prune_Parallel(int u, int v, int d, char* cache, LabelSet* device_labels, int* result, int tid){
  
  const int ws = 32;
 
  LabelSet& labels_v = device_labels[v];

  for(int i=0; i<d && (*result) == 0; i++){
    LabelSetNode& node = labels_v.array[i];

    for(int j=tid; j<node.size && (*result) == 0; j+=ws){
      int w = node.data[j];
      int cache_dist = cache[w];

      if((i + cache_dist) <= d){
	*result = 1;
	break;
      }
    }
  }

}

__device__ void GPSL_Pull_v3(int u, int d, LabelSet* device_labels,  int n, int *device_csr_row_ptr, int *device_csr_col, char* cache, char* owner, int*& new_labels, int* new_labels_size, int tid, BPLabel* device_bp, int* device_ranks, ArrayManager* array_manager, int* array_sizes, bool* should_run_prev, bool* should_run_next, int* prune_result){

  // Warp size
 const int ws = 32;

 __shared__ int global_max_rank;

 if(tid == 0){
   global_max_rank = 0;
 }

 array_sizes[tid] = 0;

 
 // Fill the cache using <d labels
 LabelSet& labels_u = device_labels[u];

 for(int i=0; i<d; i++){
  LabelSetNode& node = labels_u.array[i];

  for(int j=tid; j<node.size; j+=ws){
    cache[node.data[j]] = i;
  }

  __syncwarp();
 }

 int rank_u = device_ranks[u];

 // First we want to figure out how much memory we need to allocate
 // So we need the compute the local size for each thread
 int local_max_rank = 0;
 

 // Iterate through the neighbors
 int ngh_start = device_csr_row_ptr[u];
 int ngh_end = device_csr_row_ptr[u+1];
 for(int i=ngh_start; i<ngh_end; i++){
  int v = device_csr_col[i];
  LabelSet& labels_v = device_labels[v];
  LabelSetNode& prev_labels_v = labels_v.array[d-1];
  
  // For each neighbor check the d-1 labels in parallel
  // For each label if all the pruning stages are passed increment local size
  for(int j=0; j<prev_labels_v.size; j++){
    int w = prev_labels_v.data[j];

    if(owner[w] >= 0){
      continue;
    }

    if(cache[w] < d){ // If the vertex is already labelled with a lower distance
      continue;
    }

    int rank_w = device_ranks[w];
    if(rank_u > rank_w){ // Rank based prune
      continue;
    }

    if(GPSL_PruneByBp(u, w, d, device_bp)){ // Bit parallel prune
      continue;
    }

    if(tid == 0){
      *prune_result = 0;
    }

    // Standard prune but in parallel
    GPSL_Prune_Parallel(u,w,d,cache,device_labels,prune_result,tid);
    __syncwarp();

    if(*prune_result != 0){
      continue;
    }

    if(tid == 0){
      int selected_tid = j % ws;
      owner[w] = selected_tid;
      array_sizes[selected_tid]++;

      if(rank_w > local_max_rank){
	local_max_rank = rank_w;
      }
    }

    __syncwarp();

  }

 }

 __syncwarp();

  if(tid == 0){
   
    global_max_rank = local_max_rank;

    // Make sizes cumilative
    // So size[i-1], size[i] indicates the region thread i will write to
    for(int i=1; i<ws; i++){
      array_sizes[i] += array_sizes[i-1];
    }

    // After the previous process the last index should have the overall sum of the sizes
    int global_size = array_sizes[ws-1];

    // Allocate the memory 
    // To avoid slowness of device malloc, the memory is actually preallocated
    // So this step just distributes it
    if(global_size > 0){
      array_manager->Get(new_labels, global_size);
    } else {
      new_labels = nullptr;
    }

    // Set the size for the AddLabels kernel
    *new_labels_size = global_size;
 }

 __syncwarp();


 // Repeats the loops in the local size calculation part
 // Instead of pruning, it will just check the owner array for marks
 // If the vertex was marked by the current thread, it will be written to the array 
 int global_size = array_sizes[ws-1];
 if(global_size > 0){
  int local_size = (tid > 0) ?  array_sizes[tid] - array_sizes[tid-1] : array_sizes[0] ;
  int offset = (tid > 0) ? array_sizes[tid-1] : 0;
  int written = 0;
  
  for(int i=ngh_start; i<ngh_end; i++){

    int v = device_csr_col[i];
    LabelSet& labels_v = device_labels[v];
    LabelSetNode& prev_labels_v = labels_v.array[d-1];
  

    if(tid == 0 && !should_run_next[v] && device_ranks[v] < global_max_rank){
      should_run_next[v] = true; 
    }

    // For each neighbor check the d-1 labels in parallel
    // For each label if the label is owned by this thread, add it to the new_labels array
    for(int j=tid; j<prev_labels_v.size && written < local_size; j+=ws){

      int w = prev_labels_v.data[j];

      if(owner[w] == tid){
	new_labels[offset++] = w;
	owner[w] = -1;
	written++;
      }
    
    }

  }
 }

 __syncwarp();


 // Reset cache for <d nodes
 for(int i=0; i<d; i++){
  LabelSetNode& node = labels_u.array[i];

  for(int j=tid; j<node.size; j+=ws){
    cache[node.data[j]] = MAX_DIST;
  }
 }


}



__device__ void GPSL_Pull(int u, int d, LabelSet* device_labels,  int n, int *device_csr_row_ptr, int *device_csr_col, char* cache, char* owner, int*& new_labels, int* new_labels_size, int tid, BPLabel* device_bp, int* device_ranks, ArrayManager* array_manager, int* array_sizes, bool* should_run_prev, bool* should_run_next){

 // Warp size
 const int ws = 32;

 __shared__ int global_max_rank;
 __shared__ int cache_time;
 __shared__ int prune_time;
 __shared__ int alloc_time;
 __shared__ int write_time;

 if(tid == 0){
   global_max_rank = 0;
   cache_time = 0;
   prune_time = 0;
   alloc_time = 0;
   write_time = 0;
 }

 __syncwarp();
 
 // Fill the cache using <d labels
 LabelSet& labels_u = device_labels[u];

 int start = clock();
 for(int i=0; i<d; i++){
  LabelSetNode& node = labels_u.array[i];

  for(int j=tid; j<node.size; j+=ws){
    cache[node.data[j]] = i;
  }

  __syncwarp();
 }
 int end = clock();
 atomicMax_block(&cache_time, end-start);

 int rank_u = device_ranks[u];

 // First we want to figure out how much memory we need to allocate
 // So we need the compute the local size for each thread
 int local_size = 0;
 int local_max_rank = 0;
 

 // Iterate through the neighbors
 int ngh_start = device_csr_row_ptr[u];
 int ngh_end = device_csr_row_ptr[u+1];

 start = clock();
 for(int i=ngh_start; i<ngh_end; i++){

  int v = device_csr_col[i];
  LabelSet& labels_v = device_labels[v];
  LabelSetNode& prev_labels_v = labels_v.array[d-1];
  
  // For each neighbor check the d-1 labels in parallel
  // For each label if all the pruning stages are passed increment local size
  for(int j=tid; j<prev_labels_v.size; j+=ws){
    int w = prev_labels_v.data[j];
    int rank_w = device_ranks[w];


    if(owner[w] >= 0 || 
	cache[w] < d ||
	rank_u > rank_w ||
	GPSL_PruneByBp(u, w, d, device_bp) ||
	GPSL_Prune(u, w, d, cache, device_labels)){
      continue;
    }
   
    // Not an atomic operation but should not cause race conditions anyway
    // Because, there can't be two instances of the same label in the same neighbor
    owner[w] = tid; // Mark the vertex
    
    if(rank_w > local_max_rank)
      local_max_rank = rank_w;  
    
    local_size++; 
  }

  __syncwarp();
 
 }

 end = clock();
 atomicMax_block(&prune_time, end-start);

 // Write local size to shared memory
 array_sizes[tid] = local_size;
 atomicMax_block(&global_max_rank, local_max_rank);

 __syncwarp();

 if(tid == 0){
 
  start = clock();
 
  // Make sizes cumilative
  // So size[i-1], size[i] indicates the region thread i will write to
  for(int i=1; i<ws; i++){
    array_sizes[i] += array_sizes[i-1];
  }

  // After the previous process the last index should have the overall sum of the sizes
  int global_size = array_sizes[ws-1];


  // Allocate the memory 
  // To avoid slowness of device malloc, the memory is actually preallocated
  // So this step just distributes it
  if(global_size > 0){
    array_manager->Get(new_labels, global_size);
  } else {
    new_labels = nullptr;
  }

  // Set the size for the AddLabels kernel
  *new_labels_size = global_size;

  end = clock();
  alloc_time = end-start;
 }
 
 __syncwarp();


 // Repeats the loops in the local size calculation part
 // Instead of pruning, it will just check the owner array for marks
 // If the vertex was marked by the current thread, it will be written to the array 
 start = clock();
 int global_size = array_sizes[ws-1];
 if(global_size > 0){
  int offset = (tid > 0) ? array_sizes[tid-1] : 0;
  int written = 0;
  
  for(int i=ngh_start; i<ngh_end; i++){

    int v = device_csr_col[i];
    LabelSet& labels_v = device_labels[v];
    LabelSetNode& prev_labels_v = labels_v.array[d-1];
  

    if(tid == 0 && !should_run_next[v] && device_ranks[v] < global_max_rank){
      should_run_next[v] = true; 
    }

    // For each neighbor check the d-1 labels in parallel
    // For each label if the label is owned by this thread, add it to the new_labels array
    for(int j=tid; j<prev_labels_v.size && written < local_size; j+=ws){

      int w = prev_labels_v.data[j];

      if(owner[w] == tid){
	new_labels[offset++] = w;
	owner[w] = -1;
	written++;
      }
    
    }

  }

 }

 __syncwarp();

 end = clock();
 atomicMax_block(&write_time, end-start);


 // Reset cache for <d nodes
 for(int i=0; i<d; i++){
  LabelSetNode& node = labels_u.array[i];

  for(int j=tid; j<node.size; j+=ws){
    cache[node.data[j]] = MAX_DIST;
  }
 }

 if(tid == 0){
  printf("u=%d, cache_time=%d, prune_time=%d, alloc_time=%d, write_time=%d \n", u, cache_time, prune_time, alloc_time, write_time);
 }

   
}


__device__ void GPSL_Pull_v2(int u, int d, LabelSet* device_labels,  int n, int *device_csr_row_ptr, int *device_csr_col, char* cache, char* owner, int*& new_labels, int* new_labels_size, int tid, BPLabel* device_bp, int* device_ranks, ArrayManager* array_manager, int* array_sizes, bool* should_run_prev, bool* should_run_next, int v_size){

 // Warp size
 const int ws = 32;
 const int w_size = ws/v_size;
 const int w_tid = tid % w_size;
 const int v_group = tid / w_size;

 /* printf("v_size=%d w_size=%d w_tid=%d v_group=%d \n", v_size, w_size, w_tid, v_group); */

 __shared__ int global_max_rank;

 if(tid == 0)
   global_max_rank = 0;

 
 // Fill the cache using <d labels
 LabelSet& labels_u = device_labels[u];

 for(int i=0; i<d; i++){
  LabelSetNode& node = labels_u.array[i];

  for(int j=tid; j<node.size; j+=ws){
    cache[node.data[j]] = i;
  }

  __syncwarp();
 }

 int rank_u = device_ranks[u];

 // First we want to figure out how much memory we need to allocate
 // So we need the compute the local size for each thread
 // But since we are parallelizing at v now, there could be duplicates
 // So we first mark the vertices and then do a second loop to count
 int local_size = 0;
 int temp_size = 0;
 int local_max_rank = 0;
 

 // Iterate through the neighbors
 int ngh_start = device_csr_row_ptr[u];
 int ngh_end = device_csr_row_ptr[u+1];
 for(int i=ngh_start + v_group; i<ngh_end; i+=v_size){

  int v = device_csr_col[i];
  LabelSet& labels_v = device_labels[v];
  LabelSetNode& prev_labels_v = labels_v.array[d-1];
  
  // For each neighbor check the d-1 labels in parallel
  // For each label if all the pruning stages are passed increment temp size
  for(int j=w_tid; j<prev_labels_v.size; j+=w_size){
    int w = prev_labels_v.data[j];

    /* printf("tid=%d w=%d u=%d \n", tid, w, u); */

    if(owner[w] >= 0){ // If the vertex is marked already
      continue;
    }

    if(cache[w] < d){ // If the vertex is already labelled with a lower distance
      continue;
    }

    int rank_w = device_ranks[w];
    if(rank_u > rank_w){ // Rank based prune
      continue;
    }

    if(GPSL_PruneByBp(u, w, d, device_bp)){ // Bit parallel prune
      continue;
    }

    if(GPSL_Prune(u, w, d, cache, device_labels)){ // Standard prune
      continue;
    }
    // Not an atomic operation but should not cause race conditions anyway
    // Because, there can't be two instances of the same label in the same neighbor
    owner[w] = tid; // Mark the vertex
    
    temp_size++; 

    if(rank_w > local_max_rank)
      local_max_rank = rank_w;
  }
 
 }

 __syncwarp();

  // Count the labels for real this time
  for(int i=ngh_start + v_group; i<ngh_end && local_size < temp_size; i+=v_size){

    int v = device_csr_col[i];
    LabelSet& labels_v = device_labels[v];
    LabelSetNode& prev_labels_v = labels_v.array[d-1];
  

    // For each neighbor check the d-1 labels in parallel
    for(int j=w_tid; j<prev_labels_v.size && local_size < temp_size; j+=w_size){

      int w = prev_labels_v.data[j];

      if(owner[w] == tid){
	local_size++; // If owned count it
	owner[w] = -tid-2; // Change the mark to a distinct negative value to not count it twice
      }
    
    }

  }


  __syncwarp();


 // Write local size to shared memory
 array_sizes[tid] = local_size;
 atomicMax_block(&global_max_rank, local_max_rank);

 __syncwarp();

 if(tid == 0){
 
 
  // Make sizes cumilative
  // So size[i-1], size[i] indicates the region thread i will write to
  for(int i=1; i<ws; i++){
    array_sizes[i] += array_sizes[i-1];
  }

  // After the previous process the last index should have the overall sum of the sizes
  int global_size = array_sizes[ws-1];


  // Allocate the memory 
  // To avoid slowness of device malloc, the memory is actually preallocated
  // So this step just distributes it
  if(global_size > 0){
    array_manager->Get(new_labels, global_size);
  } else {
    new_labels = nullptr;
  }

  // Set the size for the AddLabels kernel
  *new_labels_size = global_size;
 }
 
 __syncwarp();


 // Repeats the loops in the local size calculation part
 // Instead of pruning, it will just check the owner array for marks
 // If the vertex was marked by the current thread, it will be written to the array 
 int global_size = array_sizes[ws-1];
 if(global_size > 0){
  int offset = (tid > 0) ? array_sizes[tid-1] : 0;
  int written = 0;
  
  for(int i=ngh_start + v_group; i<ngh_end; i+=v_size){

    int v = device_csr_col[i];
    LabelSet& labels_v = device_labels[v];
    LabelSetNode& prev_labels_v = labels_v.array[d-1];
  

    if(w_tid == 0 && !should_run_next[v] && device_ranks[v] < global_max_rank){
      should_run_next[v] = true; 
    }

    // For each neighbor check the d-1 labels in parallel
    // For each label if the label is owned by this thread, add it to the new_labels array
    for(int j=w_tid; j<prev_labels_v.size && written < local_size; j+=w_size){

      int w = prev_labels_v.data[j];

      if(owner[w] == -tid-2){
	new_labels[offset++] = w;
	owner[w] = -1;
	written++;
      }
    
    }

  }

 }

 __syncwarp();


 // Reset cache for <d nodes
 for(int i=0; i<d; i++){
  LabelSetNode& node = labels_u.array[i];

  for(int j=tid; j<node.size; j+=ws){
    cache[node.data[j]] = MAX_DIST;
  }
 }

   
}

__global__ void GPSL_Main_Kernel(int d, int n, LabelSet* device_labels, int* device_csr_row_ptr, int* device_csr_col, char* device_caches, char* device_owners, int** all_new_labels, int* all_new_labels_size, BPLabel* device_bp, int* device_ranks, ArrayManager* array_manager, int* array_sizes, bool* should_run_prev, bool* should_run_next, int* prune_result){

 const int ws = 32;
 const int bid = blockIdx.x;
 const int block_tid = threadIdx.x;
 const int tid = block_tid % ws;
 const int nt = blockDim.x;
 const int wid = block_tid / ws + bid * (nt / ws) ;
 const int nw = gridDim.x * (nt / ws);

 const size_t cache_offset = wid * n;
 const size_t array_sizes_offset = wid*32;

 int v_size;

 if (KERNEL_MODE == 1){
   v_size = 32;
 } else if (KERNEL_MODE == 2){
   v_size = 8;
 } else if (KERNEL_MODE == 3){
   v_size = 4;
 }

 for(int u=wid; u<n; u+=nw){
   if(d == 2 || should_run_prev[u]){
    if (KERNEL_MODE == 0){ // w-32
      GPSL_Pull(u, d, device_labels, n, device_csr_row_ptr, device_csr_col, device_caches + cache_offset, device_owners + cache_offset, all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp, device_ranks, array_manager, array_sizes + array_sizes_offset, should_run_prev, should_run_next);
    } else if (KERNEL_MODE == 4){ // hybrid w-32 and v-32
      int degree = device_csr_row_ptr[u+1] - device_csr_row_ptr[u];

      if(degree >= 256){
	GPSL_Pull_v2(u, d, device_labels, n, device_csr_row_ptr, device_csr_col, device_caches + cache_offset, device_owners + cache_offset, all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp, device_ranks, array_manager, array_sizes + array_sizes_offset, should_run_prev, should_run_next, 32);
      } else {
	GPSL_Pull(u, d, device_labels, n, device_csr_row_ptr, device_csr_col, device_caches + cache_offset, device_owners + cache_offset, all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp, device_ranks, array_manager, array_sizes + array_sizes_offset, should_run_prev, should_run_next);
      }

    } else if (KERNEL_MODE == 5){ // hybrid v-8-w-4 and v-4-w-8
      int degree = device_csr_row_ptr[u+1] - device_csr_row_ptr[u];

      if(degree >= 16){
	GPSL_Pull_v2(u, d, device_labels, n, device_csr_row_ptr, device_csr_col, device_caches + cache_offset, device_owners + cache_offset, all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp, device_ranks, array_manager, array_sizes + array_sizes_offset, should_run_prev, should_run_next, 8);
      } else {
	GPSL_Pull_v2(u, d, device_labels, n, device_csr_row_ptr, device_csr_col, device_caches + cache_offset, device_owners + cache_offset, all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp, device_ranks, array_manager, array_sizes + array_sizes_offset, should_run_prev, should_run_next, 4);
      }
    
    } else if (KERNEL_MODE == 6){
      printf("This kernel mode is not implemented yet !!! \n");

    } else if (KERNEL_MODE == 7){
	GPSL_Pull_v3(u, d, device_labels, n, device_csr_row_ptr, device_csr_col, device_caches + cache_offset, device_owners + cache_offset, all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp, device_ranks, array_manager, array_sizes + array_sizes_offset, should_run_prev, should_run_next, prune_result + wid);

    } else { // v-32 or v-8-w-4 or v-4-w-8
      GPSL_Pull_v2(u, d, device_labels, n, device_csr_row_ptr, device_csr_col, device_caches + cache_offset, device_owners + cache_offset, all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp, device_ranks, array_manager, array_sizes + array_sizes_offset, should_run_prev, should_run_next, v_size);
    }
   }
 }

}

__global__ void GPSL_Dist0(int n, int** new_labels, int* new_labels_size, int* dist0_buffer){

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  for(int u=tid; u<n; u+=nt){
    int* self_label = dist0_buffer + u;
    *self_label = u;
    new_labels[u] = self_label;
    new_labels_size[u] = 1;
  }
}

__host__ void GPSL_Dist0_Host(int n, int** new_labels, int* new_labels_size, int* dist0_buffer, int number_of_wa){
  
}


__global__ void GPSL_Dist1(int n, int* device_csr_row_ptr, int* device_csr_col, int* device_ranks, int** new_labels, int* new_labels_size, ArrayManager* array_manager, int* array_sizes, char* device_owners){

  const int ws = 32;
  const int bid = blockIdx.x;
  const int block_tid = threadIdx.x;
  const int tid = block_tid % ws;
  const int nt = blockDim.x;
  const int wid = block_tid / ws + bid * (nt / ws) ;
  const int nw = gridDim.x * (nt / ws);

  const size_t cache_offset = wid * n;
  const size_t array_sizes_offset = wid*32;
 
  char* owner = device_owners + cache_offset;
  int* sizes = array_sizes + array_sizes_offset;

  for(int u=wid; u<n; u+=nw){
    int local_size = 0;
    int ngh_start = device_csr_row_ptr[u];
    int ngh_end = device_csr_row_ptr[u+1];

    for(int i=ngh_start + tid; i < ngh_end; i += ws){
      int v = device_csr_col[i];

      if(owner[v] == -1 && device_ranks[u] < device_ranks[v]){
	owner[v] = tid;
	local_size++;
      }
    }

    sizes[tid] = local_size;
  
    __syncwarp();

    if(tid == 0){

      for(int i=1; i<ws; i++){
	sizes[i] += sizes[i-1];
      }

      int global_size = sizes[ws-1];

      if(global_size > 0)
	array_manager->Get(new_labels[u], global_size);
      else
	new_labels[u] = nullptr;
      
      new_labels_size[u] = global_size;
 
    }

    __syncwarp();
   
    int global_size = sizes[ws-1];

    if(global_size > 0){
      int written = 0;
      int offset = (tid > 0) ? sizes[tid-1] : 0;
      for(int i=ngh_start + tid; i < ngh_end && written < local_size; i += ws){
	int v = device_csr_col[i];
      
	if(owner[v] == tid){
	  new_labels[u][offset++] = v;
	  owner[v] = -1;
	  written++;
	}
      }
    }
  }
}

__global__ void GPSL_InitCache_Kernel(int n, char* device_caches, char* device_owners){
 const int ws = 32;
 const int bid = blockIdx.x;
 const int block_tid = threadIdx.x;
 const int tid = block_tid % ws;
 const int nt = blockDim.x;
 const int wid = block_tid / ws + bid * (nt / ws) ;

 int offset = wid*n;
 for(int i=tid; i<n; i+=ws){
  device_caches[offset + i] = MAX_DIST;
  device_owners[offset + i] = -1;
 }
 
}

__global__ void GPSL_InitArrayManager_Kernel(ArrayManager* array_manager, int* init_buffer){

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid == 0){
    array_manager->LoadBuffer(init_buffer);
  }

}

__global__ void GPSL_InitLabelSets_Kernel(int n, LabelSet* device_labels, LabelSetNode* device_label_nodes){
  
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  for(int u=tid; u < n; u += nt){
    int start_index = u * MAX_DIST;
    
    for(size_t i=start_index; i<start_index + MAX_DIST; i++){
      device_label_nodes[i].size = 0;
      device_label_nodes[i].data = nullptr;
    }

    device_labels[u].LoadArray(device_label_nodes + start_index);
  }
}

__global__ void GPSL_AddLabels_Kernel(int n, int d, LabelSet* device_labels, int** new_labels, int* new_labels_size, int* updated_flag){
  
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  int local_updated = 0;
  for(int u=tid; u<n; u+=nt){

    if(local_updated == 0 && new_labels_size[u] > 0){
      local_updated = 1;
    } 

    device_labels[u].Insert(new_labels[u], new_labels_size[u], d);
    new_labels[u] = nullptr;
    new_labels_size[u] = 0;
  }


  if(updated_flag != nullptr)
    atomicCAS(updated_flag, 0, local_updated);

}


__global__ void GPSL_CountLabels_Kernel(int n, int d, LabelSet* device_labels, int* device_counts){
  
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  /* printf("tid=%d, nt=%d \n", tid, nt); */

  for(int u=tid; u<n; u+=nt){
    for(int j=0; j<=d; j++){
      LabelSetNode& node = device_labels[u].array[j];
      device_counts[u + j*n] += node.size;
    }
    
  }

}

__host__ void GPSL_WriteLabelCounts(int n, int d, LabelSet* device_labels, string filename){
  
  int * device_counts;
  cudaMalloc((void**)&device_counts, sizeof(int)*n*(d+1));
  
  GPSL_CountLabels_Kernel<<<100, 32>>>(n, d, device_labels, device_counts);
  cudaDeviceSynchronize();

  int* counts = new int[n*(d+1)];
  cudaMemcpy(counts, device_counts, sizeof(int)*n*(d+1), cudaMemcpyDeviceToHost);
  
  ofstream ofs(filename);
  long long total_count = 0;
  
  for(int i=0; i<n; i++){
    ofs << i << ":\t";
    for(int j=0; j<=d; j++){
      int count = counts[i + j*n];
      ofs << count << "\t";
      total_count += count;
    }
    ofs << endl;
  }

  delete[] counts;
  cudaFree(device_counts);

  ofs << "Total Count: " << total_count << endl;
  cout << "Total Count: " << total_count << endl;
  ofs << "Avg. Count: " << total_count / ((double) n) << endl;
  cout << "Avg. Count: " << total_count / ((double) n) << endl;

  ofs.close();

}

__host__ void GPSL_Index(CSR& csr, int device, int number_of_warps, vector<BPLabel>& bp_labels, vector<int>& ranks, vector<int>& order){

  cudaSetDevice(device);

  double start_time, end_time, all_start_time, all_end_time;

  start_time = omp_get_wtime();

  size_t size;
  size_t total_size = 0;

  int* dist0_buffer;
  size = csr.n*sizeof(int);
  total_size += size;
  cudaMalloc((void**) &dist0_buffer, size);

  int* init_buffer;
  size = ARRAY_MANAGER_ALLOC_SIZE*GBS_TO_BYTES;
  total_size += size;
  cudaMalloc((void**) &init_buffer, size);  

  int* array_sizes;
  size = sizeof(int)*32*number_of_warps;
  total_size += size;
  cudaMalloc((void**) &array_sizes, size);
  cudaMemset(array_sizes, 0, size);

  int * device_csr_row_ptr; 
  size = sizeof(int)*(csr.n+1);
  total_size += size;
  cudaMalloc((void**) &device_csr_row_ptr, size);  
  cudaMemcpy(device_csr_row_ptr, csr.row_ptr, size, cudaMemcpyHostToDevice);

  int * device_csr_col; 
  size = sizeof(int)*(csr.m);
  total_size += size;
  cudaMalloc((void**) &device_csr_col, size);  
  cudaMemcpy(device_csr_col, csr.col, size, cudaMemcpyHostToDevice);

  BPLabel * device_bp; 
  size = sizeof(BPLabel)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_bp, size);  
  cudaMemcpy(device_bp, bp_labels.data(), size, cudaMemcpyHostToDevice);

  int * device_ranks;
  size = sizeof(int)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_ranks, size);  
  cudaMemcpy(device_ranks, ranks.data(), size, cudaMemcpyHostToDevice);

  char* device_caches;
  size = sizeof(char) * csr.n * (number_of_warps);
  total_size += size;
  cudaMalloc((void**) &device_caches, size);  

  char* device_owners;
  size = sizeof(char) * csr.n * (number_of_warps);
  total_size += size;
  cudaMalloc((void**) &device_owners, size);  

  LabelSet* device_labels;
  size = sizeof(LabelSet)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_labels, size);  

  LabelSetNode* device_label_nodes;
  size = sizeof(LabelSetNode)*(csr.n)*MAX_DIST;
  total_size += size;
  cudaMalloc((void**) &device_label_nodes, size);

  int** new_labels;
  size = sizeof(int*)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &new_labels, size); 
  cudaMemset(new_labels, 0, size);

  int* new_labels_size;
  size = sizeof(int)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &new_labels_size, size);  
  cudaMemset(new_labels_size, 0, size);

  int* updated_flag;
  size = sizeof(int);
  total_size += size;
  cudaMalloc((void**) &updated_flag, size);  
  cudaMemset(updated_flag, 0, size);

  ArrayManager* array_manager;
  size = sizeof(ArrayManager);
  total_size += size;
  cudaMalloc((void**) &array_manager, size);  

  bool* should_run_prev;
  size = sizeof(bool)*csr.n;
  total_size += size;
  cudaMalloc((void**)&should_run_prev, sizeof(bool)*csr.n);
  cudaMemset(should_run_prev, 0, sizeof(bool));

  bool* should_run_next;
  size = sizeof(bool)*csr.n;
  total_size += size;
  cudaMalloc((void**)&should_run_next, sizeof(bool)*csr.n);
  cudaMemset(should_run_next, 0, sizeof(bool));

  int* prune_result;
  size = sizeof(int)*number_of_warps;
  total_size += size;
  cudaMalloc((void**) &prune_result, size);
  cudaMemset(prune_result, 0, sizeof(int)*number_of_warps);
  

  end_time = omp_get_wtime();

  cout << "Memory Allocations" << ": " << end_time - start_time << " seconds" << endl;
  cout << "Total Initial Memory: " << total_size / (1024*1024) << " MB" << endl;


  start_time = omp_get_wtime();

  GPSL_InitCache_Kernel<<<number_of_warps,32>>>(csr.n, device_caches, device_owners);
  cudaDeviceSynchronize();
  GPSL_InitArrayManager_Kernel<<<1,1>>>(array_manager, init_buffer);
  cudaDeviceSynchronize();
  GPSL_InitLabelSets_Kernel<<<number_of_warps,32>>>(csr.n, device_labels, device_label_nodes);
  cudaDeviceSynchronize();

  end_time = omp_get_wtime();

  cout << "Init. Data Structures" << ": " << end_time - start_time << " seconds" << endl;


  all_start_time = omp_get_wtime();

  start_time = omp_get_wtime();

  GPSL_Dist0<<<number_of_warps,32>>>(csr.n, new_labels, new_labels_size, dist0_buffer);
  cudaDeviceSynchronize();

  GPSL_AddLabels_Kernel<<<number_of_warps,32>>>(csr.n, 0, device_labels, new_labels, new_labels_size, nullptr);
  cudaDeviceSynchronize();
  
  /* GPSL_WriteLabelCounts(csr.n, 0, device_labels, "output_gpsl_label_counts_0.txt"); */


  end_time = omp_get_wtime();
  
  cout << "Level " << "0" << ": " << end_time - start_time << " seconds" << endl;


  start_time = omp_get_wtime();
  
  GPSL_Dist1<<<number_of_warps,32>>>(csr.n, device_csr_row_ptr, device_csr_col, device_ranks, new_labels, new_labels_size, array_manager, array_sizes, device_owners);
  cudaDeviceSynchronize();
 
  GPSL_AddLabels_Kernel<<<number_of_warps,32>>>(csr.n, 1, device_labels, new_labels, new_labels_size, nullptr);
  cudaDeviceSynchronize();
 
  /* GPSL_WriteLabelCounts(csr.n, 1, device_labels, "output_gpsl_label_counts_1.txt"); */
  
  end_time = omp_get_wtime();
    
  cout << "Level " << "1" << ": " << end_time - start_time << " seconds" << endl;


  int updated = 1;
  int last_dist;
  for(int d=2; d<MAX_DIST && updated; d++){
    start_time = omp_get_wtime();

    last_dist = d;

    // Reset the updated flag
    cudaMemset(updated_flag, 0, sizeof(int));

    // Swap and reset the should_run arrays
    bool* temp = should_run_prev;
    should_run_prev = should_run_next;
    should_run_next = temp;
    cudaMemset(should_run_next, 0, sizeof(bool));

    // Run the main kernel
    GPSL_Main_Kernel<<<number_of_warps,32>>>(d, csr.n, device_labels, device_csr_row_ptr, device_csr_col, device_caches, device_owners, new_labels, new_labels_size, device_bp, device_ranks, array_manager, array_sizes, should_run_prev, should_run_next, prune_result);
    cudaDeviceSynchronize();
    
    // Add the labels to the proper places
    GPSL_AddLabels_Kernel<<<number_of_warps,32>>>(csr.n, d, device_labels, new_labels, new_labels_size, updated_flag);
    cudaDeviceSynchronize();

    // Copy the updated flag to the CPU
    cudaMemcpy(&updated, updated_flag, sizeof(int), cudaMemcpyDeviceToHost);

    end_time = omp_get_wtime();
    cout << "Level " << d << ": " << end_time - start_time << " seconds" << endl;
  }

  all_end_time = omp_get_wtime();
  
  cout << "Indexing: " << all_end_time - all_start_time << " seconds" << endl;
  
  GPSL_WriteLabelCounts(csr.n, last_dist, device_labels, "output_gpsl_label_counts_final.txt");

}


#endif
