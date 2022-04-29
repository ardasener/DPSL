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

	for(int i=0; i<4; i++){
		CSR& csr1 = *(vc4.csrs[i]);
		CSR& csr2 = *(vc4_test.csrs[i]);
	
		assert(csr1.n == csr2.n);
		assert(csr1.m == csr2.m);
		assert(csr1.row_ptr[0] == csr2.row_ptr[0]);
		assert(csr1.col[0] == csr2.col[0]);
		cout << "CSR " << i << " assertions passed" << endl;
	}

#endif
}
