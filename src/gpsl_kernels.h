#ifndef GPSL_KERNELS_H
#define GPSL_KERNELS_H

#include "common.h"
#include "bp.h"

struct ArrayManager {
  int* buffer;
  size_t last_index;
  int mutex;

  void Get(int*& ptr, int size);
  void LoadBuffer(int* new_buffer);
}

struct LabelSetNode {
  int* data;
  int size;
  LabelSetNode* next;
};

struct LabelSet {
  LabelSetNode* array;
  void Insert(int* data, int size, int dist);
  void LoadArray(LabelSetNode* new_array);
}

void GPSL_Index(CSR& csr, int device, int number_of_wraps, vector<BPLabel>* bp_labels);



#endif
