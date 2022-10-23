#ifndef GPSL_CU
#define GPSL_CU

#include "gpsl.h"
#include "omp.h"

__device__ void lock(int *mutex) {
  while (atomicCAS_block(mutex, 0, 1) != 0)
    ;
}

__device__ void unlock(int *mutex) { atomicExch_block(mutex, 0); }

__device__ void ArrayManager_Get(ArrayManager *a, int *&ptr, int size) {

  lock(&(a->mutex));

  ptr = a->buffer + a->last_index;
  a->last_index += size;

  unlock(&(a->mutex));
}

__device__ void ArrayManager_LoadBuffer(ArrayManager *a, int *new_buffer) {
  a->buffer = new_buffer;
  a->last_index = 0;
  a->mutex = 0;
}

__device__ void LabelSet_Insert(LabelSet *s, int *data, int size, int dist) {
  s->array[dist].data = data;
  s->array[dist].size = size;
}

__device__ void LabelSet_LoadArray(LabelSet *s, LabelSetNode *new_array) {
  s->array = new_array;
}

__device__ bool GPSL_Prune(int u, int v, int d, char *cache,
                           LabelSet *device_labels) {
  LabelSet &labels_v = device_labels[v];

  for (int i = 0; i < d; i++) {
    LabelSetNode &node = labels_v.array[i];

    for (int j = 0; j < node.size; j++) {
      int w = node.data[j];
      int cache_dist = cache[w];

      if ((i + cache_dist) <= d) {
        return true;
      }
    }
  }

  return false;
}

__device__ void GPSL_Pull(int u, int d, LabelSet *device_labels, int n,
                          int *device_csr_row_ptr, int *device_csr_col,
                          char *cache, char *owner, int *&new_labels,
                          int *new_labels_size, int tid, BPLabel *device_bp,
                          int *device_ranks, ArrayManager *array_manager,
                          int *array_sizes, bool *should_run_prev,
                          bool *should_run_next) {

  // Warp size
  const int ws = 32;

  __shared__ int global_max_rank;

  if (tid == 0) {
    global_max_rank = 0;
  }

  __syncwarp();

  // Fill the cache using <d labels
  LabelSet &labels_u = device_labels[u];

  for (int i = 0; i < d; i++) {
    LabelSetNode &node = labels_u.array[i];

    for (int j = tid; j < node.size; j += ws) {
      cache[node.data[j]] = i;
    }

    __syncwarp();
  }

  int rank_u = device_ranks[u];

  // First we want to figure out how much memory we need to allocate
  // So we need the compute the local size for each thread
  int local_size = 0;
  int local_max_rank = 0;

  // Iterate through the neighbors
  int ngh_start = device_csr_row_ptr[u];
  int ngh_end = device_csr_row_ptr[u + 1];

  for (int i = ngh_start; i < ngh_end; i++) {

    int v = device_csr_col[i];
    LabelSet &labels_v = device_labels[v];
    LabelSetNode &prev_labels_v = labels_v.array[d - 1];

    // For each neighbor check the d-1 labels in parallel
    // For each label if all the pruning stages are passed increment local size
    for (int j = tid; j < prev_labels_v.size; j += ws) {
      int w = prev_labels_v.data[j];
      int rank_w = device_ranks[w];

      if (owner[w] >= 0 || cache[w] < d || rank_u > rank_w ||
          GPSL_PruneByBp(u, w, d, device_bp) ||
          GPSL_Prune(u, w, d, cache, device_labels)) {
        continue;
      }

      // Not an atomic operation but should not cause race conditions anyway
      // Because, there can't be two instances of the same label in the same
      // neighbor
      owner[w] = tid; // Mark the vertex

      if (rank_w > local_max_rank)
        local_max_rank = rank_w;

      local_size++;
    }

    __syncwarp();
  }

  // Write local size to shared memory
  array_sizes[tid] = local_size;
  atomicMax_block(&global_max_rank, local_max_rank);

  __syncwarp();

  if (tid == 0) {

    // Make sizes cumilative
    // So size[i-1], size[i] indicates the region thread i will write to
    for (int i = 1; i < ws; i++) {
      array_sizes[i] += array_sizes[i - 1];
    }

    // After the previous process the last index should have the overall sum of
    // the sizes
    int global_size = array_sizes[ws - 1];

    // Allocate the memory
    // To avoid slowness of device malloc, the memory is actually preallocated
    // So this step just distributes it
    if (global_size > 0) {
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
  // If the vertex was marked by the current thread, it will be written to the
  // array
  int global_size = array_sizes[ws - 1];
  if (global_size > 0) {
    int offset = (tid > 0) ? array_sizes[tid - 1] : 0;
    int written = 0;

    for (int i = ngh_start; i < ngh_end; i++) {

      int v = device_csr_col[i];
      LabelSet &labels_v = device_labels[v];
      LabelSetNode &prev_labels_v = labels_v.array[d - 1];

      if (tid == 0 && !should_run_next[v] &&
          device_ranks[v] < global_max_rank) {
        should_run_next[v] = true;
      }

      // For each neighbor check the d-1 labels in parallel
      // For each label if the label is owned by this thread, add it to the
      // new_labels array
      for (int j = tid; j < prev_labels_v.size && written < local_size;
           j += ws) {

        int w = prev_labels_v.data[j];

        if (owner[w] == tid) {
          new_labels[offset++] = w;
          owner[w] = -1;
          written++;
        }
      }
    }
  }

  __syncwarp();

  // Reset cache for <d nodes
  for (int i = 0; i < d; i++) {
    LabelSetNode &node = labels_u.array[i];

    for (int j = tid; j < node.size; j += ws) {
      cache[node.data[j]] = MAX_DIST;
    }
  }
}

__global__ void GPSL_Main_Kernel(int d, int n, LabelSet *device_labels,
                                 int *device_csr_row_ptr, int *device_csr_col,
                                 char *device_caches, char *device_owners,
                                 int **all_new_labels, int *all_new_labels_size,
                                 BPLabel *device_bp, int *device_ranks,
                                 ArrayManager *array_manager, int *array_sizes,
                                 bool *should_run_prev, bool *should_run_next,
                                 int *prune_result) {

  const int ws = 32;
  const int bid = blockIdx.x;
  const int block_tid = threadIdx.x;
  const int tid = block_tid % ws;
  const int nt = blockDim.x;
  const int wid = block_tid / ws + bid * (nt / ws);
  const int nw = gridDim.x * (nt / ws);

  const size_t cache_offset = wid * n;
  const size_t array_sizes_offset = wid * 32;

  for (int u = wid; u < n; u += nw) {
    if (d == 2 || should_run_prev[u]) {
      GPSL_Pull(u, d, device_labels, n, device_csr_row_ptr, device_csr_col,
                device_caches + cache_offset, device_owners + cache_offset,
                all_new_labels[u], &(all_new_labels_size[u]), tid, device_bp,
                device_ranks, array_manager, array_sizes + array_sizes_offset,
                should_run_prev, should_run_next);
    }
  }
}

__global__ void GPSL_Dist0(int n, int **new_labels, int *new_labels_size,
                           int *dist0_buffer) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  for (int u = tid; u < n; u += nt) {
    int *self_label = dist0_buffer + u;
    *self_label = u;
    new_labels[u] = self_label;
    new_labels_size[u] = 1;
  }
}

__global__ void GPSL_Dist1(int n, int *device_csr_row_ptr, int *device_csr_col,
                           int *device_ranks, int **new_labels,
                           int *new_labels_size, ArrayManager *array_manager,
                           int *array_sizes, char *device_owners) {

  const int ws = 32;
  const int bid = blockIdx.x;
  const int block_tid = threadIdx.x;
  const int tid = block_tid % ws;
  const int nt = blockDim.x;
  const int wid = block_tid / ws + bid * (nt / ws);
  const int nw = gridDim.x * (nt / ws);

  const size_t cache_offset = wid * n;
  const size_t array_sizes_offset = wid * 32;

  char *owner = device_owners + cache_offset;
  int *sizes = array_sizes + array_sizes_offset;

  for (int u = wid; u < n; u += nw) {
    int local_size = 0;
    int ngh_start = device_csr_row_ptr[u];
    int ngh_end = device_csr_row_ptr[u + 1];

    for (int i = ngh_start + tid; i < ngh_end; i += ws) {
      int v = device_csr_col[i];

      if (owner[v] == -1 && device_ranks[u] < device_ranks[v]) {
        owner[v] = tid;
        local_size++;
      }
    }

    sizes[tid] = local_size;

    __syncwarp();

    if (tid == 0) {

      for (int i = 1; i < ws; i++) {
        sizes[i] += sizes[i - 1];
      }

      int global_size = sizes[ws - 1];

      if (global_size > 0)
        array_manager->Get(new_labels[u], global_size);
      else
        new_labels[u] = nullptr;

      new_labels_size[u] = global_size;
    }

    __syncwarp();

    int global_size = sizes[ws - 1];

    if (global_size > 0) {
      int written = 0;
      int offset = (tid > 0) ? sizes[tid - 1] : 0;
      for (int i = ngh_start + tid; i < ngh_end && written < local_size;
           i += ws) {
        int v = device_csr_col[i];

        if (owner[v] == tid) {
          new_labels[u][offset++] = v;
          owner[v] = -1;
          written++;
        }
      }
    }
  }
}

__global__ void GPSL_InitCache_Kernel(int n, char *device_caches,
                                      char *device_owners) {
  const int ws = 32;
  const int bid = blockIdx.x;
  const int block_tid = threadIdx.x;
  const int tid = block_tid % ws;
  const int nt = blockDim.x;
  const int wid = block_tid / ws + bid * (nt / ws);

  int offset = wid * n;
  for (int i = tid; i < n; i += ws) {
    device_caches[offset + i] = MAX_DIST;
    device_owners[offset + i] = -1;
  }
}

__global__ void GPSL_InitArrayManager_Kernel(ArrayManager *array_manager,
                                             int *init_buffer) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    ArrayManager_LoadBuffer(array_manager, init_buffer);
  }
}

__global__ void GPSL_InitLabelSets_Kernel(int n, LabelSet *device_labels,
                                          LabelSetNode *device_label_nodes) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  for (int u = tid; u < n; u += nt) {
    int start_index = u * MAX_DIST;

    for (size_t i = start_index; i < start_index + MAX_DIST; i++) {
      device_label_nodes[i].size = 0;
      device_label_nodes[i].data = nullptr;
    }

    LabelSet_LoadArray(device_labels, device_label_nodes + start_index);
  }
}

__global__ void GPSL_AddLabels_Kernel(int n, int d, LabelSet *device_labels,
                                      int **new_labels, int *new_labels_size,
                                      int *updated_flag) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nt = blockDim.x * gridDim.x;

  int local_updated = 0;
  for (int u = tid; u < n; u += nt) {

    if (local_updated == 0 && new_labels_size[u] > 0) {
      local_updated = 1;
    }

    LabelSet_Insert(&(device_labels[u]), new_labels[u], new_labels_size[u], d);
    new_labels[u] = nullptr;
    new_labels_size[u] = 0;
  }

  if (updated_flag != nullptr)
    atomicCAS(updated_flag, 0, local_updated);
}

GPSL::GPSL(CSR &csr_, int device_, int number_of_warps_, BP *bp_ptr_,
           vector<IDType> *ranks_ptr, vector<IDType> *order_ptr,
           vector<IDType> *cut_ptr_)
    : csr(csr_), device(device_), number_of_warps(number_of_warps_),
      bp_ptr(bp_ptr_), cut_ptr(cut_ptr_) {

  if (typeid(IDType) != typeid(int)) {
    throw "64-bit is not supported on GPU";
  }

  if (ranks_ptr == nullptr) {
    order = gen_order<IDType>(csr.row_ptr, csr.col, csr.n, csr.m, order_method);
    ranks.resize(csr.n);
    for (IDType i = 0; i < csr.n; i++) {
      ranks[order[i]] = i;
    }
  } else {
    ranks.insert(ranks.end(), ranks_ptr->begin(), ranks_ptr->end());
    order.insert(order.end(), order_ptr->begin(), order_ptr->end());
  }

  if (bp_ptr == nullptr) {
    bp_ptr = new BP(csr, ranks, order);
  }
}

void GPSL::Init() {

  double start_time = omp_get_wtime();

  size_t total_size = 0;
  size_t size = 0;

  size = csr.n * sizeof(int);
  total_size += size;
  cudaMalloc((void **)&device_dist0_buffer, size);

  size = sizeof(int) * 32 * number_of_warps;
  total_size += size;
  cudaMalloc((void **)&device_array_sizes, size);
  cudaMemset(device_array_sizes, 0, size);

  size = sizeof(int) * (csr.n + 1);
  total_size += size;
  cudaMalloc((void **)&device_row_ptr, size);
  cudaMemcpy(device_row_ptr, csr.row_ptr, size, cudaMemcpyHostToDevice);

  size = sizeof(int) * (csr.m);
  total_size += size;
  cudaMalloc((void **)&device_col, size);
  cudaMemcpy(device_col, csr.col, size, cudaMemcpyHostToDevice);

  size = sizeof(BPLabel) * (csr.n);
  total_size += size;
  cudaMalloc((void **)&device_bp, size);
  cudaMemcpy(device_bp, bp_ptr->bp_labels.data(), size, cudaMemcpyHostToDevice);

  size = sizeof(int) * (csr.n);
  total_size += size;
  cudaMalloc((void **)&device_ranks, size);
  cudaMemcpy(device_ranks, ranks.data(), size, cudaMemcpyHostToDevice);

  size = sizeof(char) * csr.n * (number_of_warps);
  total_size += size;
  cudaMalloc((void **)&device_caches, size);

  size = sizeof(char) * csr.n * (number_of_warps);
  total_size += size;
  cudaMalloc((void **)&device_owners, size);

  size = sizeof(LabelSet) * (csr.n);
  total_size += size;
  cudaMalloc((void **)&device_labels, size);

  size = sizeof(LabelSetNode) * (csr.n) * MAX_DIST;
  total_size += size;
  cudaMalloc((void **)&device_label_nodes, size);

  size = sizeof(int *) * (csr.n);
  total_size += size;
  cudaMalloc((void **)&device_new_labels, size);
  cudaMemset(device_new_labels, 0, size);

  size = sizeof(int) * (csr.n);
  total_size += size;
  cudaMalloc((void **)&device_new_labels_size, size);
  cudaMemset(device_new_labels_size, 0, size);

  size = sizeof(int);
  total_size += size;
  cudaMalloc((void **)&device_updated_flag, size);
  cudaMemset(device_updated_flag, 0, size);

  size = sizeof(ArrayManager);
  total_size += size;
  cudaMalloc((void **)&device_array_manager, size);

  size = sizeof(bool) * csr.n;
  total_size += size;
  cudaMalloc((void **)&device_should_run_prev, sizeof(bool) * csr.n);
  cudaMemset(device_should_run_prev, 0, sizeof(bool));

  size = sizeof(bool) * csr.n;
  total_size += size;
  cudaMalloc((void **)&device_should_run_next, sizeof(bool) * csr.n);
  cudaMemset(device_should_run_next, 0, sizeof(bool));

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  cout << "Total Memory: " << total_mem / (double)(1024 * 1024) << " MB"
       << endl;
  cout << "Free Memory: " << free_mem / (double)(1024 * 1024) << " MB" << endl;

  size_t alloc_mem = free_mem * 0.8; // Allocating 80% of the remainder
  alloc_mem -=
      (alloc_mem % 4); // Ensure it is divisible by 4 (ie, size of an integer)
  cout << "Allocating: " << alloc_mem / (double)(1024 * 1024) << " MB" << endl;

  cudaMalloc((void **)&device_init_buffer, alloc_mem);

  double end_time = omp_get_wtime();

  cout << "Memory Allocations: " << end_time - start_time << " seconds" << endl;

  start_time = omp_get_wtime();

  GPSL_InitCache_Kernel<<<number_of_warps, 32>>>(csr.n, device_caches,
                                                 device_owners);
  cudaDeviceSynchronize();
  GPSL_InitArrayManager_Kernel<<<1, 1>>>(device_array_manager,
                                         device_init_buffer);
  cudaDeviceSynchronize();
  GPSL_InitLabelSets_Kernel<<<number_of_warps, 32>>>(csr.n, device_labels,
                                                     device_label_nodes);
  cudaDeviceSynchronize();

  end_time = omp_get_wtime();

  cout << "Init. Data Structures"
       << ": " << end_time - start_time << " seconds" << endl;
}

void GPSL::Level0(int **new_labels, int *new_labels_size) {

  start_time = omp_get_wtime();

  GPSL_Dist0<<<number_of_warps, 32>>>(csr.n, new_labels, new_labels_size,
                                      dist0_buffer);
  cudaDeviceSynchronize();

  end_time = omp_get_wtime();

  cout << "Level "
       << "0"
       << ": " << end_time - start_time << " seconds" << endl;
}

void GPSL::Level1(int **new_labels, int *new_labels_size) {
  start_time = omp_get_wtime();

  GPSL_Dist1<<<number_of_warps, 32>>>(
      csr.n, device_csr_row_ptr, device_csr_col, device_ranks, new_labels,
      new_labels_size, array_manager, array_sizes, device_owners);
  cudaDeviceSynchronize();

  end_time = omp_get_wtime();

  cout << "Level "
       << "1"
       << ": " << end_time - start_time << " seconds" << endl;
}

void GPSL::Index() {
  Level0();
  Level1();
}

#endif
