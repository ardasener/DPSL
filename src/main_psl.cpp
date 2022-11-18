#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "psl/psl.h"
#include "utils/common.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: <exe> graph_file" << endl;
    return 1;
  }

  string filename(argv[1]);

  cout << "Reading graph..." << endl;
  CSR csr(filename);

  cout << "PSL Preprocessing..." << endl;
  string order_method = ORDER_METHOD;
  PSL psl(csr, order_method);

  cout << "PSL Indexing..." << endl;
  psl.Index();
  //psl.WriteLabelCounts("output_psl_label_counts.txt");
  cout << "PSL Querying..." << endl;
  psl.QueryTest(5);
  // psl.Query(12, "output_psl_query.txt");
}
