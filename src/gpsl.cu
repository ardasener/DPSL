#ifndef GPSL_CU
#define GPSL_CU

#include "common.h"
#include "bp.h"
#include "external/order/order.hpp"

#define LINKED_LIST_NODE_SIZE 16


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
  int data[LINKED_LIST_NODE_SIZE];
  LinkedListNode* next;
};

struct LinkedList {
  LinkedListNode* head = nullptr;
  LinkedListNode* tail = nullptr;
  int size = 0;
  int last_index = 0;
  
  __host__ __device__ void Insert(int data){

    if(tail == nullptr){
      tail = new LinkedListNode;
      head = tail;
      tail->next = nullptr;
      last_index = 0;
    }
    
    if(last_index == LINKED_LIST_NODE_SIZE){
      tail->next = new LinkedListNode;
      tail = tail->next;
      tail->next = nullptr;
      last_index = 0;
    }

    tail->data[last_index++] = data;

    size++;
  }

  __host__ __device__ ~LinkedList() {
    LinkedListNode* ptr = head;
    while(ptr != nullptr){
      LinkedListNode* temp = ptr;
      ptr = ptr->next;
      delete temp;
    }
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
    for(int i=0; i<node->size; i++){
      int w = node->data[i];
      int cache_dist = cache[w];

      if((i + cache_dist) <= d){
        return true;
      }
    }
    node = node->next;
    dist++;
  }

  return false;
}

__device__ void GPSL_Pull(int u, int d, LabelSet* device_labels,  int n, int *device_csr_row_ptr, int *device_csr_col, char* cache, int*& new_labels, int* new_labels_size, int tid, BPLabel* device_bp, int* device_ranks){

 const int ws = 32;
 __shared__ int global_size;
 __shared__ int sizes[ws];

 if(tid == 0){
  global_size = 0;

  for(int i=0; i<ws; i++){
    sizes[i] = 0;
  }
 } 

 __syncwarp();

 LabelSet& labels_u = device_labels[u];

 LabelSetNode* node = labels_u.head;
 int dist = 0;
 while(node != nullptr){
  for(int i=tid; i<node->size; i+=ws){
    cache[node->data[i]] = dist;
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
  for(int j=tid; j<labels_v.tail->size; j+=ws){
    int w = labels_v.tail->data[j];

    /* printf("tid=%d w=%d u=%d \n", tid, w, u); */

    if(cache[w] == d)
      continue;

    if(device_ranks[u] > device_ranks[w])
      continue;

    if(GPSL_PruneByBp(u, w, d, device_bp))
      continue;

    if(GPSL_Prune(u, w, d, cache, device_labels))
      continue;
  
   
    local_label_list.Insert(w);          
    cache[w] = d;
  }
 
 }


 // Compute the total number of labels
 int local_size = local_label_list.size;
 atomicAdd_block(&global_size, local_size);
 sizes[tid] = local_size;

 __syncwarp();

 // Allocate the necessary memory
 if(tid == 0){
  
  /* printf("Dist=%d, u=%d, global_size=%d \n", d, u, global_size); */
  
  new_labels = new int[global_size]; 

  if(new_labels == nullptr){
    printf("Nullptr at u=%d \n", u);
  }

  *new_labels_size = global_size;

  // Make sizes cummilative
  for(int i=1; i<ws; i++){
    sizes[i] += sizes[i-1];
  }
 }
 
 __syncwarp();

 // Convert & combine the linked lists into an array
 LinkedListNode* list_node = local_label_list.head;
 int offset = (tid > 0) ? sizes[tid-1] : 0;
 while(list_node != nullptr){
   
   int last_index_node = LINKED_LIST_NODE_SIZE;
   
   if(list_node->next == nullptr){
    last_index_node = local_label_list.last_index;
   }
  
   for(int i=0; i<last_index_node; i++){
    int label = list_node->data[i];
    new_labels[offset++] = label;
   }
   
   list_node = list_node->next;
 }

 __syncwarp();


 // Reset cache for <d nodes
 node = labels_u.head;
 while(node != nullptr){
  for(int i=tid; i<node->size; i+=ws){
    cache[node->data[i]] = MAX_DIST;
  }
  node = node->next;
 }

 // Reset cache for =d nodes
 for(int i=tid; i<global_size; i+=ws){
  cache[new_labels[i]] = MAX_DIST;
 }


   
}

__global__ void GPSL_Main_Kernel(int d, int n, LabelSet* device_labels, int* device_csr_row_ptr, int* device_csr_col, char** device_caches, int** all_new_labels, int* all_new_labels_size, BPLabel* device_bp, int* device_ranks){

 const int ws = 32;
 const int bid = blockIdx.x;
 const int block_tid = threadIdx.x;
 const int tid = block_tid % ws;
 const int nt = blockDim.x;
 const int wid = block_tid / ws + bid * (nt / ws) ;
 const int nw = gridDim.x * (nt / ws);


 for(int u=wid; u<n; u+=nw){
   GPSL_Pull(u, d, device_labels, n, device_csr_row_ptr, device_csr_col, device_caches[wid], all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp, device_ranks);
 }

}

__global__ void GPSL_Dist0(int n, int** new_labels, int* new_labels_size){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  for(int u=tid; u<n; u+=nt){
    int* self_label = new int;
    *self_label = u;
    new_labels[u] = self_label;
    new_labels_size[u] = 1;
  }
}


__global__ void GPSL_Dist1(int n, int* device_csr_row_ptr, int* device_csr_col, int** new_labels, int* new_labels_size){
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  for(int u=tid; u<n; u+=nt){
    int csr_start = device_csr_row_ptr[u];
    int csr_end = device_csr_row_ptr[u+1];
    int size = csr_end - csr_start;

    int* neighbor_labels = new int[size];
    for(int i=csr_start; i<csr_end; i++){
      int v = device_csr_col[i];
      neighbor_labels[i - csr_start] = v;
    }
  
    new_labels[u] = neighbor_labels;
    new_labels_size[u] = size;
  }
}

__global__ void GPSL_InitCache_Kernel(int n, char** device_caches){
 const int ws = 32;
 const int bid = blockIdx.x;
 const int block_tid = threadIdx.x;
 const int tid = block_tid % ws;
 const int nt = blockDim.x;
 const int wid = block_tid / ws + bid * (nt / ws) ;
 const int nw = gridDim.x * (nt / ws);


 if(tid == 0){
  device_caches[wid] = new char[n];
 }

 __syncwarp();

 for(int i=tid; i<n; i+=ws){
  device_caches[wid][i] = MAX_DIST;
 }
 
}

__global__ void GPSL_AddLabels_Kernel(int n, LabelSet* device_labels, int** new_labels, int* new_labels_size, int* updated_flag){
  
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  for(int u=tid; u<n; u+=nt){

    if(updated_flag != nullptr && *updated_flag == 0 && new_labels_size[u] > 0){
      atomicExch(updated_flag, 1);
    }

    device_labels[u].Insert(new_labels[u], new_labels_size[u]);
    new_labels[u] = nullptr;
    new_labels_size[u] = 0;
  }

}


__global__ void GPSL_CountLabels_Kernel(int n, LabelSet* device_labels, int* device_counts){
  
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  /* printf("tid=%d, nt=%d \n", tid, nt); */

  for(int u=tid; u<n; u+=nt){
    int local_count = 0;
    LabelSetNode* ptr = device_labels[u].head;
    
    while(ptr != nullptr){
      local_count += ptr->size;
      ptr = ptr->next;
    }

    device_counts[u] = local_count;
  }


}

__host__ void GPSL_WriteLabelCounts(int n, LabelSet* device_labels, string filename){
  
  int * device_counts;
  cudaMalloc((void**)&device_counts, sizeof(int)*n);
  
  GPSL_CountLabels_Kernel<<<32, 100>>>(n, device_labels, device_counts);
  cudaDeviceSynchronize();

  int* counts = new int[n];
  cudaMemcpy(counts, device_counts, sizeof(int)*n, cudaMemcpyDeviceToHost);
  
  ofstream ofs(filename);
  long long total_count = 0;
  
  for(int i=0; i<n; i++){
    ofs << i << " : " << counts[i] << endl;
    total_count += counts[i];
  }

  delete[] counts;
  cudaFree(device_counts);

  ofs << "Total Count: " << total_count << endl;
  ofs << "Avg. Count: " << total_count / ((double) n) << endl;

  ofs.close();

}

__host__ void GPSL_Index(CSR& csr, int number_of_warps){

  // CUDA has a limit on the heapsize by default when allocating memory inside a kernel
  // This part increases that substantially
  size_t heapsize = 1ULL << 32;
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapsize);

  vector<int> order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, "degree");

  vector<int> ranks(csr.n);
  for(int i=0; i<csr.n; i++){
    ranks[order[i]] = i;
  }

  BP bp(csr, ranks, order);

  int size;
  int total_size = 0;

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
  cudaMemcpy(device_bp, bp.bp_labels.data(), size, cudaMemcpyHostToDevice);


  int * device_ranks;
  size = sizeof(int)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_ranks, size);  
  cudaMemcpy(device_ranks, ranks.data(), size, cudaMemcpyHostToDevice);

  char** device_caches;
  size = sizeof(char*)*(number_of_warps);
  total_size += size;
  cudaMalloc((void**) &device_caches, size);  

  LabelSet* device_labels;
  size = sizeof(LabelSet)*(csr.n);
  total_size += size;
  cudaMalloc((void**) &device_labels, size);  

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
  

  GPSL_InitCache_Kernel<<<number_of_warps,32>>>(csr.n, device_caches);
  cudaDeviceSynchronize();

  double start_time, end_time, all_start_time, all_end_time;

  all_start_time = omp_get_wtime();

  start_time = omp_get_wtime();

  GPSL_Dist0<<<number_of_warps,32>>>(csr.n, new_labels, new_labels_size);

  cudaDeviceSynchronize();

  GPSL_AddLabels_Kernel<<<number_of_warps,32>>>(csr.n, device_labels, new_labels, new_labels_size, nullptr);
  
  cudaDeviceSynchronize();
  
  GPSL_WriteLabelCounts(csr.n, device_labels, "output_gpsl_label_counts_0.txt");
  
  cudaDeviceSynchronize();
  
  GPSL_Dist1<<<number_of_warps,32>>>(csr.n, device_csr_row_ptr, device_csr_col, new_labels, new_labels_size);
  
  cudaDeviceSynchronize();
 
  GPSL_AddLabels_Kernel<<<number_of_warps,32>>>(csr.n, device_labels, new_labels, new_labels_size, nullptr);

  cudaDeviceSynchronize();
  
  GPSL_WriteLabelCounts(csr.n, device_labels, "output_gpsl_label_counts_1.txt");

  cudaDeviceSynchronize();

  end_time = omp_get_wtime();
    
  cout << "Level " << "0&1" << ": " << end_time - start_time << " seconds" << endl;

  int updated = 1;
  for(int d=2; d<MAX_DIST && updated; d++){
    start_time = omp_get_wtime();

    cudaMemset(updated_flag, 0, size);
    
    GPSL_Main_Kernel<<<number_of_warps,32>>>(d, csr.n, device_labels, device_csr_row_ptr, device_csr_col, device_caches, new_labels, new_labels_size, device_bp, device_ranks);
    cudaDeviceSynchronize();
    
    GPSL_AddLabels_Kernel<<<number_of_warps,32>>>(csr.n, device_labels, new_labels, new_labels_size, updated_flag);
    cudaDeviceSynchronize();

    cudaMemcpy(&updated, updated_flag, sizeof(int), cudaMemcpyDeviceToHost);

    end_time = omp_get_wtime();
    cout << "Level " << d << ": " << end_time - start_time << " seconds" << endl;
  }

  all_end_time = omp_get_wtime();
  
  cout << "Indexing: " << all_end_time - all_start_time << " seconds" << endl;
  
  GPSL_WriteLabelCounts(csr.n, device_labels, "output_gpsl_label_counts_final.txt");

}

__host__ int main(int argc, char* argv[]){
  CSR csr(argv[1]);
  GPSL_Index(csr, 100);
}

#endif
