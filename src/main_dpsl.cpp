#include "dpsl/dpsl.h"
#include "mpi.h"
#include "utils/common.h"
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cerr << "Usage 1 : <exe> graph_file partion_file" << endl;
    cerr << "Usage 2 : <exe> graph_file partitioner_binary [partition_params]"
         << endl;
    return 1;
  }

  string filename(argv[1]);
  string partition(argv[2]);
  string partition_params = "";

  if (argc >= 4) {
    partition_params = argv[3];
  }

  MPI_Init(&argc, &argv);
  int pid, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  cout << "PID=" << pid << endl;

  if (pid == 0) {
    cout << "Reading " << filename << "..." << endl;
    CSR csr(filename);
    cout << "Number of Processes:" << np << endl;
    DPSL dpsl(pid, &csr, np, partition, partition_params);
    dpsl.Index();
    // dpsl.WriteLabelCounts("output_dpsl_label_counts.txt");
    dpsl.QueryTest(5);
    // dpsl.Query(12, "output_dpsl_query.txt");

  } else {
    DPSL dpsl(pid, nullptr, np, "", "");
    dpsl.Index();
    // dpsl.WriteLabelCounts("");
    dpsl.QueryTest(5);
    // dpsl.Query(12, "");
  }
  MPI_Finalize();
}