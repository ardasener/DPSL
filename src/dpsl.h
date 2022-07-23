#ifndef DPSL_H
#define DPSL_H

#include "common.h"
#include "cut.h"
#include "mpi.h"
#include "psl.h"
#include <algorithm>
#include <fstream>
#include <ostream>
#include <string>
#include <unordered_set>
#include <set>

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
  MPI_CSR_NODES,
  MPI_CUT,
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
};

class DPSL {

public:
  template <typename T>
  void SendData(T *data, int size, int vertex, int to,
                MPI_Datatype type = MPI_IDType);

  template <typename T>
  void BroadcastData(T *data, int size, int vertex,
                     MPI_Datatype type = MPI_IDType);

  template <typename T>
  int RecvData(T *&data, int vertex, int from, MPI_Datatype type = MPI_IDType);

  bool MergeCut(vector<vector<IDType> *> new_labels, PSL &psl);
  void Barrier();
  int pid, np;
  IDType global_n;
  CSR *whole_csr = nullptr;
  CSR *part_csr = nullptr;
  BP *global_bp = nullptr;
  IDType *partition;
  vector<IDType> cut;
  vector<bool> in_cut;
  vector<IDType> ranks;
  vector<IDType> order;
  VertexCut *vc_ptr = nullptr;
  int last_dist;
  char **caches;
  void InitP0(string vsep_file="");
  void Init();
  void Index();
  void WriteLabelCounts(string filename);
  void Query(IDType u, string filename);
  void Log(string msg);
  void PrintTime(string tag, double time);
  PSL *psl_ptr;
  DPSL(int pid, CSR *csr, int np, string vsep_file = "");
  ~DPSL();
};

#endif
