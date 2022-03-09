#include "dpsl.h"
#include "external/order/order.hpp"
#include <iostream>

using namespace std;

int main(int argc, char* argv[]){

	cout << "Reading graph..." << endl;
	CSR csr(argv[1]);

	cout << "Ordering vertices..." << endl;
	vector<int> order;
	tie(order, csr.row_ptr, csr.col, csr.n, csr.m) 
		= gen_order(csr.row_ptr, csr.col, csr.n, csr.m, "degree");

	cout << "Ranking vertices..." << endl;
	vector<int> ranks(csr.n);
	for(int i=0; i<csr.n; i++){
		ranks[order[i]] = i;
	}

	cout << "PSL Preprocessing..." << endl;
	PSL psl(csr, ranks);

	cout << "PSL Indexing..." << endl;
	psl.Index();
	psl.WriteLabelCounts("output_label_counts.txt");
	
	cout << "PSL Querying..." << endl;
	psl.Query(11, "output_query.txt");
}
