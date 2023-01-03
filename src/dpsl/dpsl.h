#ifndef DPSL_H
#define DPSL_H

#include <algorithm>
#include <fstream>
#include <ostream>
#include <set>
#include <string>
#include <unordered_set>

#include "../psl/psl.h"
#include "../utils/common.h"
#include "cut.h"
#include "mpi.h"

using namespace std;

#ifdef USE_64_BIT
const MPI_Datatype MPI_IDType = MPI_INT64_T;
#else
const MPI_Datatype MPI_IDType = MPI_INT32_T;
#endif

enum MPI_CONSTS {
  MPI_GLOBAL_N,
  MPI_CSR_ROW_PTR,
  MPI_CSR_COL,
  MPI_CSR_IDS,
  MPI_CSR_INV_IDS,
  MPI_CSR_TYPE,
  MPI_CSR_COMP_IDS,
  MPI_CUT,
  MPI_CUT_MERGE_ORDER,
  MPI_PARTITION,
  MPI_CACHE,
  MPI_UPDATED,
  MPI_BP_DIST,
  MPI_BP_SET,
  MPI_BP_USED,
  MPI_LABEL_INDICES,
  MPI_LABELS,
  MPI_VERTEX_RANKS,
  MPI_VERTEX_ORDER,
  MPI_LEAF_ROOT,
  MPI_LOCAL_MIN
};

class DPSL {
 public:
  template <typename T>
  void SendData(T *data, size_t size, int tag, int to,
                MPI_Datatype type = MPI_IDType);

  template <typename T>
  void BroadcastData(T *data, size_t size, MPI_Datatype type = MPI_IDType);

  template <typename T>
  size_t RecvData(T *&data, int tag, int from, MPI_Datatype type = MPI_IDType);

  template <typename T>
  size_t RecvBroadcast(T *&data, int from, MPI_Datatype type = MPI_IDType);

  size_t CompressCutLabels(IDType *&comp_indices, IDType *&comp_labels,
                           vector<vector<IDType> *> &new_labels,
                           size_t start_index, size_t end_index);
  size_t MergeCut(vector<vector<IDType> *> &new_labels, PSL &psl);

  void Barrier();
  int pid, np;
  IDType global_n;
  CSR *whole_csr = nullptr;
  CSR *unordered_csr = nullptr;
  CSR *part_csr = nullptr;
  BP *global_bp = nullptr;
  IDType *partition;
  vector<IDType> cut;
  vector<IDType> cut_merge_order;
  vector<bool> in_cut;
  vector<bool> local_min;
  vector<IDType> leaf_root;
  vector<IDType> ranks;
  vector<IDType> order;
  vector<vector<bool>> merge_seen;
  VertexCut *vc_ptr = nullptr;
  int last_dist;
  char **caches;
  vector<bool> *used;
  void InitP0(string partition_str = "", string partition_params = "");
  void Init();
  void Index();
  void WriteLabelCounts(string filename);
  bool Query(IDType u, string filename, bool verbose = true);
  void QueryTest(int query_count);
  void Log(string msg);
  void PrintTime(string tag, double time);
  PSL *psl_ptr;
  DPSL(int pid, CSR *csr, int np, string partition_str = "",
       string partition_params = "");
  ~DPSL();
};

#endif
