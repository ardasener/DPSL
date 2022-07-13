#include "dpsl.h"
#include "psl.h"
#include "external/order/order.hpp"
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include "mpi.h"

using namespace std;


void RunPSL(string filename){
	cout << "Reading graph..." << endl;
	CSR csr(filename);

	cout << "__PSL__" << endl;

	cout << "PSL Preprocessing..." << endl;
	string order_method = ORDER_METHOD;
	PSL psl(csr, order_method);

	cout << "PSL Indexing..." << endl;
	psl.Index();
	psl.WriteLabelCounts("output_psl_label_counts.txt");
	
	cout << "PSL Querying..." << endl;
	psl.Query(11, "output_psl_query.txt");
}

void RunDPSL(int pid, int np, string filename, string vsep_filename){
	
	if(pid == 0){
		cout << "Reading " << filename << "..." << endl;
		CSR csr(filename);
		cout << "Number of Processes:" << np << endl;
		DPSL dpsl(pid, &csr, np, vsep_filename);
		dpsl.Index();
		dpsl.WriteLabelCounts("output_dpsl_label_counts.txt");
		dpsl.Query(11, "output_dpsl_query.txt");
	} else {
		DPSL dpsl(pid, nullptr, np, "");
		dpsl.Index();
		dpsl.WriteLabelCounts("");
		dpsl.Query(11, "");
	}
	
}

int main(int argc, char* argv[]){

	if(argc < 2){
		cerr << "Usage: <exe> graph_file [part_file (for DPSL only)]" << endl;
		return 1;
	}
	
	string filename(argv[1]);

	string vsep_filename = "";
	bool mpi = false;
	if(argc >= 3){
		vsep_filename = argv[2];
		mpi = true;
	}

	bool use64bit = false;
	if(MPI_IDType == MPI_INT64_T)
		use64bit = true;

	cout << "GRAPH=" << filename	
		<< " PARTITION=" << vsep_filename	
		<< " MPI=" << mpi 
		<< " NUM_THREADS=" << NUM_THREADS
		<< " N_ROOTS=" << N_ROOTS
		<< " RERANK_CUT=" << RERANK_CUT
		<< " GLOBAL_RANKS=" << GLOBAL_RANKS
		<< " LOCAL_BP=" << USE_LOCAL_BP
		<< " GLOBAL_BP=" << USE_GLOBAL_BP
		<< " ORDER_METHOD=" << ORDER_METHOD
		<< " 64_BIT=" << use64bit
		<< endl;

	if(!mpi){
		RunPSL(filename);
	} else {
		MPI_Init(&argc, &argv);
		int pid, np;
		MPI_Comm_rank(MPI_COMM_WORLD,&pid);
		MPI_Comm_size(MPI_COMM_WORLD,&np);
		RunDPSL(pid, np, filename, vsep_filename);
		MPI_Finalize();
	}

}
