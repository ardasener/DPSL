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

// Number of roots used for BP
#ifndef N_ROOTS
#define N_ROOTS 15
#endif

// Method used for ordering (ranking) of the vertices
#ifndef ORDER_METHOD
#define ORDER_METHOD "degree"
#endif

// Whether the ranks of vertices on the cut should be artifically increased
// This is required for the DPSL algorithm to work
#ifndef RERANK_CUT
#define RERANK_CUT true
#endif

// Whether global or local ranks should be used in DPSL
// Currently only local ranks are tested
#ifndef GLOBAL_RANKS
#define GLOBAL_RANKS false
#endif

// Each node gets an internal BP
#ifndef USE_LOCAL_BP
#define USE_LOCAL_BP false
#endif

// A single BP is created for all nodes
#ifndef USE_GLOBAL_BP
#define USE_GLOBAL_BP false
#endif

// Number of threads to be used in OpenMP sections
#ifndef NUM_THREADS
#define NUM_THREADS 16
#endif

// Scheduling strategy to be used in OpenMP sections
#ifndef SCHEDULE
#define SCHEDULE dynamic,256
#endif

// Experimental feature !!!
// If != 0, vertices with less labels then this value will be processed without distance cache
#ifndef SMART_DIST_CACHE_CUTOFF
#define SMART_DIST_CACHE_CUTOFF 0
#endif

// Experimental feature !!!
// Allows duplicates in BP
#ifndef ALLOW_DUPLICATE_BP
#define ALLOW_DUPLICATE_BP false
#endif

// Experimental feature !!!
// Uses a dynamic rank system for BP construction
#ifndef BP_RERANK
#define BP_RERANK false
#endif

// Eliminates local minimum nodes 
// (Similar to the optimization in PSL*)
#ifndef ELIM_MIN
#define ELIM_MIN false
#endif

// Compresses the graph by removing, identical nodes 
// (Similar to the optimization in PSL+)
// On DPSL, this causes the compression to happen on each node separetely
#ifndef LOCAL_COMPRESS
#define LOCAL_COMPRESS false
#endif

// Compresses the graph by removing, identical nodes 
// (Similar to the optimization in PSL+)
// On DPSL, this causes the compression to happen on each node separetely
#ifndef GLOBAL_COMPRESS
#define GLOBAL_COMPRESS false
#endif

// Eliminates leaf nodes (degree 1 nodes)
#ifndef ELIM_LEAF
#define ELIM_LEAF false
#endif

// Performs an early prune operation by keeping track of the maximum ranks
// Vertices which would never pull any labels based on their rank will not attempt pull at all
#ifndef MAX_RANK_PRUNE
#define MAX_RANK_PRUNE true
#endif

// Maximum message size for MPI sections
#define MAX_COMM_SIZE 1<<30

// Sizes of the chunks during the merge algorithm of DPSL
// If set to a high number, more memory will be used during merge
// If set to a low number, the merge operation might take longer
#define MERGE_CHUNK_SIZE 1000000

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

  // Given an index in the CSR, gives the ID number of the vertex at that index
  IDType *ids = nullptr; 
  // Given the ID number of a vertex, gives the index of that vertex in the CSR
  IDType *inv_ids = nullptr; 
  // Stores the type of the vertex (indexed with real_ids)
  // 0 indicates the vertex is still in the graph
  // 1 indicates the vertex is compressed out, but has an edge to its root (ie, can be reached with 1 hop)
  // 2 indicates the vertex is compressed out, and has no edge to its root (ie, requires 2 hops to reach)
  char* type = nullptr;

  ~CSR();
  CSR(CSR& csr);
  CSR(IDType * row_ptr, IDType *col, IDType* ids, IDType* inv_ids, char* type, IDType n, IDType m);
  CSR(string filename);
  void Reorder(vector<IDType>& order, vector<IDType>* cut = nullptr, vector<bool>* in_cut = nullptr);
  void Sort();
  void InitIds();
  void Compress(vector<bool>& in_cut);
  void ComputeF1F2(vector<size_t>& f1, vector<size_t>& f2);
};


void WriteStats(const vector<Stats>& stats_vec, string filename);
vector<int>* BFSQuery(CSR& csr, IDType u);
size_t random_range(const size_t & min, const size_t & max);

#endif
