#ifndef CUT_H
#define CUT_H

#include "../utils/common.h"
#include "../external/order/order.hpp"
#include "omp.h"
#include <cmath>
#include <vector>

using namespace std;

class VertexCut {
public:
  unordered_set<IDType> cut;
  vector<CSR *> csrs;
  IDType *partition = nullptr;
  vector<IDType> ranks;
  vector<IDType> order;

  static VertexCut *Partition(CSR &csr, string partitioner, string params,
                              string order_method, int np);
  static VertexCut *Read(CSR &csr, string part_file, string order_method,
                         int np);
  void Init(CSR& csr, int np);
  ~VertexCut();
};

#endif
