#include "dpsl.h"
#include "psl.h"
#include "external/order/order.hpp"
#include "external/toml/toml.h"
#include <cstddef>
#include <iostream>
#include "metis.h"
#include <string>
#include <vector>
#include "mpi.h"

using namespace std;


void RunPSL(const toml::Value& config, string filename){
	cout << "Reading graph..." << endl;
	CSR csr(filename);

	cout << "__PSL__" << endl;

	cout << "PSL Preprocessing..." << endl;
	string order_method = config.find("order_method")->as<string>();
	PSL psl(csr, order_method);

	cout << "PSL Indexing..." << endl;
	psl.Index();
	psl.WriteLabelCounts("output_psl_label_counts.txt");
	
	cout << "PSL Querying..." << endl;
	psl.Query(11, "output_psl_query.txt");
}

void RunDPSL(const toml::Value& config, int pid, int np, string filename){
	
	string order_method = config.find("order_method")->as<string>();
	
	if(pid == 0){
		cout << "Reading " << filename << "..." << endl;
		CSR csr(filename);
		cout << "Number of Processes:" << np << endl;
		DPSL dpsl(pid, &csr, config ,np);
		dpsl.Index();
		dpsl.WriteLabelCounts("output_dpsl_label_counts.txt");
		dpsl.Query(11, "output_dpsl_query.txt");
	} else {
		DPSL dpsl(pid, nullptr, config, np);
		dpsl.Index();
		dpsl.WriteLabelCounts("");
		dpsl.Query(11, "");
	}
	
}

int main(int argc, char* argv[]){

	if(argc < 2){
		cerr << "Usage: <exe> config_file graph_file" << endl;
		return 1;
	}

	std::ifstream ifs(argv[1]);
	toml::ParseResult pr = toml::parse(ifs);
	if (!pr.valid()) {
		cout << pr.errorReason << endl;
		return 1;
	}
	
	const toml::Value& config = pr.value;
	string filename(argv[2]);

	if(!(config.find("mpi")->as<bool>())){
		RunPSL(config, filename);
	} else {
		MPI_Init(&argc, &argv);
		int pid, np;
		MPI_Comm_rank(MPI_COMM_WORLD,&pid);
		MPI_Comm_size(MPI_COMM_WORLD,&np);
		RunDPSL(config, pid, np, filename);
		MPI_Finalize();
	}

}
