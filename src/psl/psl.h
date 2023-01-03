#ifndef PSL_H
#define PSL_H

#include <omp.h>
#include <stdio.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../external/order/order.hpp"
#include "../external/pigo/pigo.hpp"
#include "../utils/bp.h"
#include "../utils/common.h"

using namespace std;

// Stores the labels for each vertex
struct LabelSet {
  // IDs of the vertices in order of distances
  vector<IDType> vertices; 
  // Indices for the start and end of distances for the vertices vector
  vector<IDType> dist_ptrs; 
};

enum PruneIDs {
  PRUNE_RANK,
  PRUNE_GLOBAL_BP,
  PRUNE_LOCAL_BP,
  PRUNE_LABEL,
};

class PSL {
 public:
  CSR &csr;
  CSR *unordered_csr = nullptr;
  vector<IDType> ranks;
  vector<IDType> order;
  BP *local_bp = nullptr;
  BP *global_bp = nullptr;
  int last_dist = 2;
  long long prune_rank = 0;
  long long prune_labels = 0;
  long long prune_local_bp = 0;
  long long prune_global_bp = 0;
  vector<long long> cand_counts;

  void ConstructBPLabel();

  vector<LabelSet> labels;
  vector<Stats> stats_vec;
  vector<IDType> max_ranks;
  vector<IDType> prev_max_ranks;
  vector<bool> in_cut;
  vector<bool> local_min;
  vector<IDType> leaf_root;
  char **caches = nullptr;
  vector<bool> *used = nullptr;
  PSL(CSR &csr_, string order_method, vector<IDType> *cut = nullptr,
      BP *global_bp = nullptr, vector<IDType> *ranks_ptr = nullptr,
      vector<IDType> *order_ptr = nullptr);
  ~PSL();
  vector<IDType> *Init(IDType u);
  void Index();
  void WriteLabelCounts(string filename);
  vector<IDType> *Query(IDType u);
  void CountPrune(int i);

  template <bool use_cache = true>
  bool Prune(IDType u, IDType v, int d, char *cache);

  template <bool use_cache = true>
  vector<IDType> *Pull(IDType u, int d, char *cache, vector<bool> &used_vec);

#ifdef GPU
  void setDevice(int device);
  void LoadGPU(CSR &csr);
  void LoadGPU(BP &bp);
  void LoadGPU(vector<LabelSet> &labels, int d);
  vector<vector<IDType> *> PullGPU(vector<IDType> &vertices, int d);
#endif

  void Query(IDType u, string filename);
  void QueryTest(int query_count);
  void CountStats(double time);
};

#endif
