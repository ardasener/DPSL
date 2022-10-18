#include "common.h"
#include "external/order/order.hpp"
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "dpsl.h"
#include "mpi.h"


int main(int argc, char* argv[]){
    if(argc < 3){
		cerr << "Usage 1 : <exe> graph_file partion_file" << endl;
		cerr << "Usage 2 : <exe> graph_file partitioner_binary [partition_params]" << endl;
		return 1;
	}

    string filename(argv[1]);
    string partition(argv[2]);
    string partition_params = "";

	if(argc >= 4){
	    partition_params = argv[2];
	}

    MPI_Init(&argc, &argv);
    int pid, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    cout << "PID=" << pid << endl;

    if(pid == 0){
        cout << "Reading " << filename << "..." << endl;
        CSR csr(filename);
        cout << "Number of Processes:" << np << endl;
        DPSL dpsl(pid, &csr, np, partition, partition_params);
        dpsl.Index();
        dpsl.WriteLabelCounts("output_dpsl_label_counts.txt");
        dpsl.QueryTest(5);
        dpsl.Query(103496, "output_dpsl_query.txt"); // Writes to file only in Debug mode

    } else {
        DPSL dpsl(pid, nullptr, np, "", "");
        dpsl.Index();
        dpsl.WriteLabelCounts(""); // Only P0 writes to the file, so filename is empty here
        dpsl.QueryTest(5);
        dpsl.Query(103496, ""); // Again only P0 writes to the file
    }
    MPI_Finalize();


}