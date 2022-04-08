#ifndef COMMON_H
#define COMMON_H

#include "external/pigo/pigo.hpp"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <iostream>

#define N_ROOTS 16
/* #define MAX_BP_THREADS 1 */
#define USE_LOCAL_BP true
#define USE_GLOBAL_BP false
#define NUM_THREADS 16


using namespace std;

using PigoCOO = pigo::COO<int, int, int *, true, false, true, false,
                          float, float *>;

const char MAX_DIST = CHAR_MAX;

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};



struct CSR {
  int *row_ptr;
  int *col;
  int n;
  int m;

  CSR(CSR& csr){
    row_ptr = new int[csr.n+1];
    col = new int[csr.m];

    for(int i=0; i<csr.n+1; i++){
      row_ptr[i] = csr.row_ptr[i];
    }

    for(int i=0; i<csr.m; i++){
      col[i] = csr.col[i];
    }

    n = csr.n;
    m = csr.m;

  }

  CSR(int * row_ptr, int *col, int n, int m): 
    row_ptr(row_ptr), col(col), n(n), m(m) {}

  CSR(string filename) {
    PigoCOO pigo_coo(filename);

    int *coo_row = pigo_coo.x();
    int *coo_col = pigo_coo.y();
    m = pigo_coo.m();
    n = pigo_coo.n();
    cout << "N:" << n << endl;
    cout << "M:" << m << endl;

    int min1 = *min_element(coo_row, coo_row+m, less<int>());
    int min2 = *min_element(coo_col, coo_col+m, less<int>());
    int min = (min1 < min2) ? min1 : min2;

    if(min != 0){
      cout << "Fixing indices with minimum=" << min << endl;
#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
      for(int i=0; i<m; i++){
        coo_row[i] -= min;
        coo_col[i] -= min;
      }
    }


    vector<pair<int,int>> edges(m);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for(int i=0; i<m; i++){
      edges[i] = make_pair(coo_row[i], coo_col[i]);
    }

    sort(edges.begin(), edges.end(), less<pair<int, int>>());

    auto unique_it = unique(edges.begin(), edges.end(), [](const pair<int,int>& p1, const pair<int,int>& p2){
	    return (p1.first == p2.first) && (p1.second == p2.second);
	  });

    edges.erase(unique_it, edges.end());

    m = edges.size();
    cout << "Unique M:" << m << endl;

    row_ptr = new int[n + 1];
    col = new int[m];

    fill(row_ptr, row_ptr+n+1, 0);

    for (int i = 0; i < m; i++) {
      col[i] = edges[i].second;
      row_ptr[edges[i].first]++;
    }

    for (int i = 1; i <= n; i++) {
      row_ptr[i] += row_ptr[i - 1];
    }

    for (int i = n; i > 0; i--) {
      row_ptr[i] = row_ptr[i - 1];
    }

    row_ptr[0] = 0;


    delete[] coo_row;
    delete[] coo_col;
  }
};

vector<int>* BFSQuery(CSR& csr, int u){

  vector<int>* dists = new vector<int>(csr.n, -1);
  auto& dist = *dists;

	int q[csr.n];

	int q_start = 0;
	int q_end = 1;
	q[q_start] = u;

	dist[u] = 0;
	while(q_start < q_end){
		int curr = q[q_start++];

		int start = csr.row_ptr[curr];
		int end = csr.row_ptr[curr+1];

		for(int i=start; i<end; i++){
		      int v = csr.col[i];

		      if(dist[v] == -1){
			      dist[v] = dist[curr]+1;

			      q[q_end++] = v;
		      }
	      }

	}

  return dists;
}

#endif
