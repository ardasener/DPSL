#ifndef PSL_H
#define PSL_H

#include "common.h"
#include "external/order/order.hpp"
#include "external/pigo/pigo.hpp"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <unordered_set>
#include <unordered_map>
#include "bp.h"


using namespace std;



// Stores the labels for each vertex
struct LabelSet {
  vector<IDType> vertices; // global vertex ids in order of distances - log(n) 
  vector<IDType> dist_ptrs; // indices for the vertices vector denoting distance starts - max_dist
};

enum PruneIDs {
  PRUNE_RANK,
  PRUNE_GLOBAL_BP,
  PRUNE_LOCAL_BP,
  PRUNE_LABEL,
};


class PSL {

public:
  
  CSR& csr;
  vector<IDType> ranks;
  vector<IDType> order;
  BP* local_bp = nullptr;
  BP* global_bp = nullptr;
  int last_dist = 2;
  long long prune_rank = 0;
  long long prune_labels = 0;
  long long prune_local_bp = 0;
  long long prune_global_bp = 0;

  void ConstructBPLabel();

  vector<LabelSet> labels;
  vector<Stats> stats_vec;
  vector<IDType> max_ranks;
  vector<bool> in_cut;
  char** caches = nullptr;
  bool** used = nullptr;
  PSL(CSR &csr_, string order_method, vector<IDType>* cut=nullptr, BP* global_bp=nullptr, vector<IDType>* ranks_ptr=nullptr, vector<IDType>* order_ptr = nullptr);
  ~PSL();
  vector<IDType>* Pull(IDType u, int d, char* cache);
  vector<IDType>* Init(IDType u);
  void Index();
  void WriteLabelCounts(string filename);
  vector<IDType>* Query(IDType u);
  void CountPrune(int i);
  bool Prune(IDType u, IDType v, int d, char* cache);
  void Query(IDType u, string filename);
  void QueryTest(int query_count);
  void CountStats(double time);
};


#endif
