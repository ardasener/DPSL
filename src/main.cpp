#include "dpsl.h"
#include "external/order/order.hpp"
#include <iostream>

using namespace std;

int main(){
	CSR csr("small_graphs/soc-dolphins.mtx");
	vector<int> order;
	tie(order, csr.row_ptr, csr.col, csr.n, csr.m) 
		= gen_order(csr.row_ptr, csr.col, csr.n, csr.m, "degree");

	vector<int> ranks(csr.n);

	for(int i=0; i<csr.n; i++){
		ranks[order[i]] = i;
	}

	PSL psl(csr, ranks);
	psl.Index();

	vector<int> res = psl.Query(11);

	for(int i=0; i<csr.n; i++){
		cout << i << " " << res[i] << endl;
	}
}
