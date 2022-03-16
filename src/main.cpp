#include "dpsl.h"
#include "psl.h"
#include "external/order/order.hpp"
#include "external/toml/toml.h"
#include <cstddef>
#include <iostream>
#include <metis.h>
#include <string>
#include <vector>
#include "mpi.h"

using namespace std;


void RunPSL(const toml::Value& config){
	cout << "Reading graph..." << endl;
	string filename = config.find("filename")->as<string>();
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

void RunDPSL(const toml::Value& config, int pid, int np){
	
	string filename = config.find("filename")->as<string>();
	string order_method = config.find("order_method")->as<string>();
	
	if(pid == 0){
		cout << "Reading " << filename << "..." << endl;
		CSR csr(filename);
		cout << "Number of Processes:" << np << endl;
		DPSL dpsl(pid, &csr, config ,np);
		dpsl.Index();
	} else {
		DPSL dpsl(pid, nullptr, config, np);
		dpsl.Index();
	}
	
}

int main(int argc, char* argv[]){

	std::ifstream ifs(argv[1]);
	toml::ParseResult pr = toml::parse(ifs);
	if (!pr.valid()) {
		cout << pr.errorReason << endl;
		return 1;
	}
	const toml::Value& config = pr.value;


	if(!(config.find("mpi")->as<bool>())){
		RunPSL(config);
	} else {
		MPI_Init(&argc, &argv);
		int pid, np;
		MPI_Comm_rank(MPI_COMM_WORLD,&pid);
		MPI_Comm_size(MPI_COMM_WORLD,&np);
		RunDPSL(config, pid, np);
		MPI_Finalize();
	}

}
