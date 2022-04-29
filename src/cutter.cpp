#include "cut.h"
#include <iostream>
#include <assert.h>
#include "omp.h"

using namespace std;

#define TEST_CUT


int main(int argc, char* argv[]){

	if(argc < 3){
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
	CSR csr(filename);

	double start, end;

	cout << "Cutting k=2 ..." << endl;
	start = omp_get_wtime();
	VertexCut vc2(csr, "degree", 2, config);
	end = omp_get_wtime();
	cout << "DONE " << end-start << " seconds" << endl;

	cout << "Cutting k=4 ..." << endl;
	start = omp_get_wtime();
	VertexCut vc4(csr, "degree", 4, config);
	end = omp_get_wtime();
	cout << "DONE " << end-start << " seconds" << endl;

	cout << "Writing the cuts to file..." << endl;
	start = omp_get_wtime();
	string filename_part2 = filename + ".part2";
	string filename_part4 = filename + ".part4";
	vc2.Write(filename_part2);
	vc4.Write(filename_part4);
	end = omp_get_wtime();
	cout << "DONE " << end-start << " seconds" << endl;

#ifdef TEST_CUT
	cout << "Reading the cut from file..." << endl;
	start = omp_get_wtime();
	VertexCut vc4_test(filename_part4);
	end = omp_get_wtime();
	cout << "DONE " << end-start << " seconds" << endl;

	cout << "Performing assertions..." << endl;

	assert(vc4_test.cut.size() == vc4.cut.size());
	cout << "Cut size assertion passed" << endl;

	for(int u : vc4_test.cut){
		auto it = vc4.cut.find(u);
		assert(it != vc4.cut.end());
	}
	cout << "Cut data assertion passed" << endl;

	int n,m;
	for(int i=0; i<4; i++){
		CSR& csr1 = *(vc4.csrs[i]);
		CSR& csr2 = *(vc4_test.csrs[i]);
		
		n = csr1.n;
		m = csr1.m;

		assert(csr1.n == csr2.n);
		assert(csr1.m == csr2.m);
		assert(csr1.row_ptr[0] == csr2.row_ptr[0]);
		assert(csr1.row_ptr[n/2] == csr2.row_ptr[n/2]);
		assert(csr1.row_ptr[n] == csr2.row_ptr[n]);
		assert(csr1.col[0] == csr2.col[0]);
		assert(csr1.col[m/2] == csr2.col[m/2]);
		assert(csr1.col[m-1] == csr2.col[m-1]);
		cout << "CSR " << i << " assertions passed" << endl;
	}

	assert(vc4.partition[0] == vc4_test.partition[0]);
	assert(vc4.partition[n/2] == vc4_test.partition[n/2]);
	assert(vc4.partition[n-1] == vc4_test.partition[n-1]);
	cout << "Partition assertions passed" << endl;

	
	assert(vc4.ranks[0] == vc4_test.ranks[0]);
	assert(vc4.ranks[n/2] == vc4_test.ranks[n/2]);
	assert(vc4.ranks[n-1] == vc4_test.ranks[n-1]);
	cout << "Rank assertions passed" << endl;

	assert(vc4.order[0] == vc4_test.order[0]);
	assert(vc4.order[n/2] == vc4_test.order[n/2]);
	assert(vc4.order[n-1] == vc4_test.order[n-1]);
	cout << "Order assertions passed" << endl;


#endif

	return 0;
}
