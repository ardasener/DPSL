#ifndef COMMON_H
#define COMMON_H

#include "external/pigo/pigo.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <limits>

#ifndef N_ROOTS
#define N_ROOTS 16
#endif

#ifndef ORDER_METHOD
#define ORDER_METHOD "b_cent"
#endif

#ifndef RERANK_CUT
#define RERANK_CUT true
#endif

#ifndef GLOBAL_RANKS
#define GLOBAL_RANKS false
#endif

#ifndef USE_LOCAL_BP
#define USE_LOCAL_BP false
#endif

#ifndef USE_GLOBAL_BP
#define USE_GLOBAL_BP false
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 16
#endif

#ifndef SCHEDULE
#define SCHEDULE dynamic,256
#endif

#ifndef SMART_DIST_CACHE_CUTOFF
#define SMART_DIST_CACHE_CUTOFF 0
#endif

#ifndef ALLOW_DUPLICATE_BP
#define ALLOW_DUPLICATE_BP false
#endif

#ifndef BP_RERANK
#define BP_RERANK false
#endif

#define MPI_CHUNK_SIZE 1<<30

using namespace std;


#ifdef USE_64_BIT
using IDType = int64_t;
const IDType MAX_ID = numeric_limits<int64_t>::max();
#else
using IDType = int32_t;
const IDType MAX_ID = numeric_limits<int32_t>::max();
#endif

using PigoCOO = pigo::COO<IDType, IDType, IDType *, true, false, true, false,
                          float, float *>;

const char MAX_DIST = CHAR_MAX;

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

struct Stats {
  long long prune_rank;
  long long prune_labels;
  long long prune_local_bp;
  long long prune_global_bp;
  long long total_label_count;
  double avg_label_count;
  double time_elapsed;
};


struct CSR {
  IDType *row_ptr;
  IDType *col;
  IDType n;
  IDType m;
  IDType *real_ids = nullptr; 
  IDType *reorder_ids = nullptr; 

  ~CSR();
  CSR(CSR& csr);
  CSR(IDType* row_ptr, IDType *col, IDType n, IDType m);
  CSR(string filename);
  void Reorder(vector<IDType>& order, vector<IDType>* cut = nullptr, vector<bool>* in_cut = nullptr);
};


void WriteStats(const vector<Stats>& stats_vec, string filename);
vector<int>* BFSQuery(CSR& csr, IDType u);
size_t random_range(const size_t & min, const size_t & max);

#endif
