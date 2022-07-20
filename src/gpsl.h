#ifndef GPSL_H
#define GPSL_H

#include "bp.h"
#include "common.h"

struct ArrayManager {
  int* buffer;
  size_t last_index;
  int mutex;
};

struct LabelSetNode {
  int* data;
  int size;
};

struct LabelSet {
  LabelSetNode* array;
};


struct GPSL {
  // Input Parameters
  CSR& csr;
  int device;
  int number_of_warps;
  
  // Device pointers
  int* device_init_buffer;
  int* device_dist0_buffer;
  int* device_array_sizes;
  int* device_row_ptr;
  int* device_col;
  BPLabel* device_bp;
  int* device_ranks;
  char* device_caches;
  char* device_owners;
  LabelSet* device_labels;
  LabelSetNode* device_label_nodes;
  int** device_new_labels;
  int* device_new_labels_size;
  int* device_updated_flag;
  ArrayManager* device_array_manager;
  bool* device_should_run_prev;
  bool* device_should_run_next;
 
  GPSL(CSR& csr_, int device_, int number_of_warps_): csr(csr_), device(device_), number_of_warps(number_of_warps_) {}
  void Init(); 
  void Index(); 
};

#endif
