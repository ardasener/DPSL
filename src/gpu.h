#ifndef GPSL_H
#define GPSL_H

#include "common.h"
#include "bp.h"


struct LabelSetNode {
  int* data;
  int size;
  LabelSetNode* next;
};

struct LabelSet {
  LabelSetNode* head = nullptr;
  LabelSetNode* tail = nullptr;
  
  __host__ __device__ void Insert(int * data, int size){
    LabelSetNode* new_node = new LabelSetNode;
    new_node->data = data;
    new_node->size = size;
    new_node->next = nullptr;

    if(head == nullptr){
      head = new_node;
      tail = new_node;
    } else {
      tail->next = new_node;
      tail = tail->next;
    }
  }
};

struct LinkedListNode {
  int data;
  LinkedListNode* next;
};

struct LinkedList {
  LinkedListNode* head = nullptr;
  int size = 0;
  
  __host__ __device__ void Insert(int data){
    LinkedListNode* new_node = new LinkedListNode;
    new_node->data = data;
    new_node->next = nullptr;

    if(head == nullptr){
      head = new_node;
      tail = new_node;
    } else {
      tail->next = new_node;
      tail = tail->next;
    }

    size++;
  }
};

__device__ void lock(int* mutex){
  while(atomicCAS_block(mutex, 0, 1) != 0);
}

__device__ void unlock(int* mutex){
  atomicExch_block(mutex, 0);
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

__device__ bool GPSL_Prune(int u, int v, int d, char* cache, LabelSet* device_labels){
  LabelSet& labels_v = device_labels[v];

  LabelSetNode* node = labels_v.head;

  int dist = 0;
  while(node != nullptr){
    for(int i=0; i<node.size; i++){
      int w = node.data[i];
      int cache_dist = cache[w];

      if((i + cache_dist) <= d){
        return true;
      }
    }
    node = node->next;
    dist++;
  }
}

__device__ void GPSL_Pull(int u, int d, LabelSet* device_labels, char* cache, int n, int *device_csr_row_ptr, int *device_csr_col, int*& new_labels, int* new_labels_size, int tid, BPLabel* device_bp, int* device_ranks){

 const int ws = 32;
 __shared__ int global_size = 0;
 __shared__ int mutex = 0;
 __shared__ int last_index = 0;

 LabelSet& labels_u = device_labels[u];

 LabelSetNode* node = labels_u.head;
 int dist = 0;
 while(node != nullptr){
  for(int i=tid; i<node.size; i+=ws){
    cache[node.vertices[i]] = dist;
  }

  dist++;
  node = node->next;
 }

 LinkedList local_label_list;
  
 int ngh_start = device_csr_row_ptr[u];
 int ngh_end = device_csr_col[u+1];
 for(int i=ngh_start; i<ngh_end; i++){
  int v = device_csr_col[i];

  LabelSet& labels_v = device_labels[v];
  if(labels_v.tail != nullptr){
    for(int j=tid; j<labels_v.tail.size; j+=ws){
      int w = labels_v.tail.vertices[j];

      if(cache[w] == d)
        continue;

      if(ranks[u] > ranks[w])
        continue;

      if(GPSL_PruneByBp(u, w, d, device_bp))
        continue;

      if(GPSL_Prune(u, w, d, cache, device_labels))
        continue;
      
      local_label_list.Insert(w);          
      atomicExch_block(cache + w, d);
    }
  }
   
 }


 // Compute the total number of labels
 int local_size = local_label_list.size;
 atomicAdd_block(&global_size, local_size);
 
 __syncwarp();

 // Allocate the necessary memory
 if(tid == 0){
  new_labels = new int[global_size]; 
  *new_labels_size = global_size;
 }
 
 __syncwarp();

 // Convert & combine the linked lists into an array
 lock(mutex);
 LinkedListNode* list_node = local_label_list.head;
 while(list_node != nullptr){
  new_labels[last_index++] = list_node.data;
  list_node = list_node->next;
 }
 unlock(mutex);

 __syncwarp();

 // Reset cache for <d nodes
 node = labels_u.head;
 while(node != nullptr){
  for(int i=tid; i<node.size; i+=ws){
    cache[node.vertices[i]] = MAX_DIST;
  }
  node = node->next;
 }

 // Reset cache for =d nodes
 for(int i=tid; i<last_index; i+=ws){
  cache[new_labels[i]] = MAX_DIST;
 }


   
}

__global__ GPSL_Kernel(int d, int n, LabelSet* device_labels, int* device_csr_row_ptr, int* device_csr_col, char** device_caches, int** all_new_labels, int* all_new_labels_size, BPLabel* device_bp, int* device_ranks){

 const int bid = blockIdx.x;
 const int block_tid = threadIdx.x;
 const int tid = block_tid % ws;
 const int nt = blockDim.x;
 const int wid = block_tid / ws + bid * (nt / ws) 
 const int nw = gridDim.x * (nt / ws);

 for(int u=wid; u<n; u+=nw){
   int * new_labels;
   int * new_labels_size;
   GPSL_Pull(u, d, n, device_labels, device_csr_row_ptr, device_csr_col, device_caches[wid], new_labels, new_labels_size, tid, device_bp, int* device_ranks);
   all_new_labels[i] = new_labels;
   all_new_labels_size[i] = new_labels_size;
 }

}


class GPSL {
public:
  // HOST DATA
  CSR* host_csr;
  BP* host_bp;
  int* ranks;

  // GLOBAL DATA
  int n;
  int m;

  // DEVICE DATA
  int* device_csr_row_ptr;
  int* device_csr_col;
  char** device_caches;
  BPLabel* device_bp;
  int* device_ranks;
  LabelSet* device_labels;

  
  __device__ void Pull(int u);

};



#endif
