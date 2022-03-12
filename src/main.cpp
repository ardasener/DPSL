#include "psl.h"
#include "external/order/order.hpp"
#include "external/toml/toml.h"
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
	
	string filename = config.find("general.filename")->as<string>();
	string order_method = config.find("general.order_method")->as<string>();

	CSR* mycsr;
	vector<int> cut;

	// Process 0 cuts the graph and sends it to other processes
	if(pid == 0){
		CSR csr(filename);

		VertexCut vc(csr, order_method, np);

		cut.insert(cut.begin(),vc.cut.begin(), vc.cut.end());
		auto& ranks = vc.ranks;

		sort(cut.begin(), cut.end(), [ranks](int u, int v){
			return ranks[u] > ranks[v];
		});

		mycsr = vc.csrs[0];

		for(int p=1; p<np; p++){
			CSR& csr2 = *(vc.csrs[p]);
			MPI_Send(&csr2.n, 1, MPI_INT32_T, p, MPI_CSR_N, MPI_COMM_WORLD);
			MPI_Send(&csr2.m, 1, MPI_INT32_T, p, MPI_CSR_M, MPI_COMM_WORLD);
		}

		for(int p=1; p<np; p++){
			CSR& csr2 = *(vc.csrs[p]);
			MPI_Send(csr2.row_ptr, csr.n+1, MPI_INT32_T, p, MPI_CSR_ROW_PTR, MPI_COMM_WORLD);
			MPI_Send(csr2.col, csr.m, MPI_INT32_T, p, MPI_CSR_COL, MPI_COMM_WORLD);
			MPI_Send(cut.data(), csr.n, MPI_INT32_T, p, MPI_CUT, MPI_COMM_WORLD);
			delete vc.csrs[p];
		}

	// Other processes take the data
	} else {
		int n, m;
		MPI_Recv(&n, 1, MPI_INT32_T, 0, MPI_CSR_N, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&m, 1, MPI_INT32_T, 0, MPI_CSR_M, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		int *row_ptr = new int[n+1];
		int *col = new int[m];
		cut.resize(n);

		MPI_Recv(row_ptr, n+1, MPI_INT32_T, 0, MPI_CSR_ROW_PTR, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(col, m, MPI_INT32_T, 0, MPI_CSR_COL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(cut.data(), n, MPI_INT32_T, 0, MPI_CUT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		mycsr = new CSR(row_ptr, col, n, m);
	}


	PSL psl(*mycsr, order_method, &cut);
	CSR& csr = *mycsr;

	psl.Init();

	if (pid == 0){
		
		vector<vector<int>> sizes(np, vector<int>(cut.size(), 0));
		for(int p=1; p<np; p++){
			MPI_Recv(&(sizes[p]), cut.size(), MPI_INT32_T, p, MPI_LABELS_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		
		for(int i=0; i<cut.size(); i++){
			auto& labels = psl.labels[cut[i]];
			for(int p=0; p<np; p++){
				int size = sizes[p][i];
				vector<int> new_labels(size);
				MPI_Recv(new_labels.data(), size, MPI_INT32_T, p, MPI_LABELS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				auto unique_it = unique(new_labels.begin(), new_labels.end());
				labels.vertices.insert(labels.vertices.end(), unique_it, new_labels.end());
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}


	} else {
		vector<int> sizes(cut.size());
		for(int i=0; i<cut.size(); i++){
			sizes[i] = psl.labels[cut[i]].vertices.size();
		}
		MPI_Send(sizes.data(), cut.size(), MPI_INT32_T, 0, MPI_LABELS_SIZE, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		for(int i=0; i<cut.size(); i++){
			int u = cut[i];
			auto& labels = psl.labels[u];
			
			MPI_Send(labels.vertices.data(), sizes[i], MPI_INT32_T, 0, MPI_LABELS, MPI_COMM_WORLD);
		}
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


	if(config.find("general.mpi")->as<bool>()){
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
