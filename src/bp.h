#ifndef BP_H
#define BP_H

#include "common.h"

using namespace std;

enum Mode {
  GLOBAL_BP_MODE,
  LOCAL_BP_MODE
};

class BP {
public:
    vector<uint64_t> bp_0_sets;
    vector<uint64_t> bp_1_sets;
    vector<uint8_t> bp_dists;
    vector<bool> used;
  
    BP(CSR& csr, vector<IDType>* cut = nullptr, Mode mode=GLOBAL_BP_MODE);
    BP(vector<uint64_t>& bp_0_sets, vector<uint64_t>& bp_1_sets, vector<uint8_t>& bp_dists, vector<bool>& used);
    void InitBPForRoot(IDType r, vector<IDType>& Sr, int root_index, CSR& csr);
    bool PruneByBp(IDType u, IDType v, int d);
    int QueryByBp(IDType u, IDType v);
    void ReorderBP(CSR& csr);
};

#endif
