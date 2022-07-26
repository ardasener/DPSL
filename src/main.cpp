#include "common.h"
#include "external/order/order.hpp"
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#ifdef BIN_PSL
#include "psl.h"
#endif

#ifdef BIN_DPSL
#include "dpsl.h"
#include "mpi.h"
#endif

#ifdef BIN_GPSL
#include "gpsl.h"
#endif

using namespace std;


int main(int argc, char* argv[]){

	if(argc < 2){
		cerr << "Usage: <exe> graph_file [part_file (for DPSL only)]" << endl;
		return 1;
	}
	
	string filename(argv[1]);

	string vsep_filename = "";
	if(argc >= 3){
		vsep_filename = argv[2];
	}

	bool use64bit = false;

#ifdef USE_64_BIT
	use64bit = true;
#endif

	cout << "GRAPH=" << filename	
		<< " PARTITION=" << vsep_filename	
		<< " NUM_THREADS=" << NUM_THREADS
		<< " N_ROOTS=" << N_ROOTS
		<< " RERANK_CUT=" << RERANK_CUT
		<< " GLOBAL_RANKS=" << GLOBAL_RANKS
		<< " LOCAL_BP=" << USE_LOCAL_BP
		<< " GLOBAL_BP=" << USE_GLOBAL_BP
		<< " ORDER_METHOD=" << ORDER_METHOD
		<< " 64_BIT=" << use64bit
		<< endl;


#ifdef BIN_PSL
	cout << "Reading graph..." << endl;
	CSR csr(filename);

	cout << "PSL Preprocessing..." << endl;
	string order_method = ORDER_METHOD;
	PSL psl(csr, order_method);

	cout << "PSL Indexing..." << endl;
	psl.Index();
	psl.WriteLabelCounts("output_psl_label_counts.txt");
	cout << "PSL Querying..." << endl;
	psl.Query(11, "output_psl_query.txt"); // Writes to file only in Debug mode
#endif

#ifdef BIN_DPSL 
		MPI_Init(&argc, &argv);
		int pid, np;
		MPI_Comm_rank(MPI_COMM_WORLD,&pid);
		MPI_Comm_size(MPI_COMM_WORLD,&np);
		
		if(pid == 0){
			cout << "Reading " << filename << "..." << endl;
			CSR csr(filename);
			cout << "Number of Processes:" << np << endl;
			DPSL dpsl(pid, &csr, np, vsep_filename);
			dpsl.Index();
			dpsl.WriteLabelCounts("output_dpsl_label_counts.txt");
			dpsl.Query(11, "output_dpsl_query.txt"); // Writes to file only in Debug mode
	
		} else {
			DPSL dpsl(pid, nullptr, np, "");
			dpsl.Index();
			dpsl.WriteLabelCounts(""); // Only P0 writes to the file, so filename is empty here
			dpsl.Query(11, ""); // Again only P0 writes to the file
		}
		MPI_Finalize();
#endif
			
#ifdef BIN_GPSL
	cout << "Reading " << filename << "..." << endl;
	CSR csr(filename);
	GPSL gpsl(csr, 0, 100);
	gpsl.Init();
#endif

}
