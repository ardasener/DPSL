#ifndef GPSL_H
#define GPSL_H

#include "../utils/bp.h"
#include "../utils/common.h"
#include "cut.h"
#include <typeid>

struct ArrayManager {
  int *buffer;
  size_t last_index;
  int mutex;
};

struct LabelSetNode {
  int *data;
  int size;
  int dist;
};

struct LabelSet {
  LabelSetNode *array;
};

struct GPSL {
  // Input Parameters
  CSR &csr;
  int device;
  int number_of_warps;

  // Device Variables
  int *device_init_buffer;
  int *device_dist0_buffer;
  int *device_array_sizes;
  int *device_row_ptr;
  int *device_col;
  BPLabel *device_bp;
  int *device_ranks;
  char *device_caches;
  char *device_owners;
  LabelSet *device_labels;
  LabelSetNode *device_label_nodes;
  int **device_new_labels;
  int *device_new_labels_size;
  int *device_updated_flag;
  ArrayManager *device_array_manager;
  bool *device_should_run_prev;
  bool *device_should_run_next;

  // Host Variables
  BP *bp_ptr;
  VertexCut *cut_ptr;
  vector<int> order;
  vector<int> ranks;

  GPSL(CSR &csr_, int device_, int number_of_warps_, BP *bp_ptr_ = nullptr,
       vector<IDType> *ranks_ptr = nullptr, vector<IDType> *order_ptr = nullptr,
       vector<IDType> *cut_ptr_ = nullptr);
  void Init();
  void Level0(int **new_labels, int *new_label_sizes);
  void Level1(int **new_labels, int *new_label_sizes);
  void LevelN(int d, int **new_labels, int *new_label_sizes);
  void AddLabels(int **labels, int *label_sizes);
  void Index();
};

#endif
