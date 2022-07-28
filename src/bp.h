#ifndef BP_H
#define BP_H

#include "common.h"

using namespace std;

struct BPLabel{
  uint8_t bp_dists[N_ROOTS];
  uint64_t bp_sets[N_ROOTS][2];
};

enum Mode {
  GLOBAL_BP_MODE,
  LOCAL_BP_MODE
};

class BP {
public:
    vector<BPLabel> bp_labels;
    vector<bool> used;
  
    BP(CSR& csr, vector<IDType>& ranks, vector<IDType>& order, vector<IDType>* cut = nullptr, Mode mode=GLOBAL_BP_MODE);
    BP(vector<BPLabel>& bp_labels, vector<bool>& used);
    void InitBPForRoot(IDType r, vector<IDType>& Sr, int root_index, CSR& csr);
    bool PruneByBp(IDType u, IDType v, int d);
    int QueryByBp(IDType u, IDType v);
    void ReorderBP(CSR& csr);
};

#endif
