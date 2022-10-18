#ifndef CUT_H
#define CUT_H

#include "common.h"
#include "external/order/order.hpp"
#include <vector>
#include "omp.h"
#include <cmath>

using namespace std;

class VertexCut{
public:
  unordered_set<IDType> cut;
  vector<CSR*> csrs;
  IDType* partition = nullptr;
  vector<IDType> ranks;
  vector<IDType> order;

  static VertexCut* Partition(CSR& csr, string partitioner, string params, string order_method, int np);
  static VertexCut* Read(CSR& csr, string part_file, string order_method, int np);
  ~VertexCut();
};

#endif
