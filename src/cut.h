#ifndef CUT_H
#define CUT_H

#include "common.h"
#include "external/order/order.hpp"
#include <vector>
#include "omp.h"
#include <cmath>

using namespace std;

class VertexCut{
public:
  unordered_set<IDType> cut;
  vector<CSR*> csrs;
  IDType* partition = nullptr;
  vector<IDType> ranks;
  vector<IDType> order;

  // VertexCut(CSR& csr, string order_method, int np, const toml::Value& config);
  VertexCut(CSR& csr, string part_file, string order_method, int np);
  ~VertexCut();
};

/*
inline VertexCut::VertexCut(CSR& csr, string vsep_file, string order_method) {
 
  cout << "Ordering..." << endl;
  order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  cout << "Ranking..." << endl;
  ranks.resize(csr.n);
  for(IDType i=0; i<csr.n; i++){
    ranks[order[i]] = i;
  }

  int np = 2;

  ifstream ifs(vsep_file);

  partition = new IDType[csr.n];
  IDType part_num;
  for(IDType i=0; i<csr.n; i++){
    ifs >> part_num;
    partition[i] = part_num;
    
    if(part_num == 2){
      cut.insert(i);
    }
  }

  cout << "Calculating edges and nodes..." << endl;
  vector<vector<pair<IDType,IDType>>> edges(np);
  for(IDType u=0; u<csr.n; u++){
    IDType start = csr.row_ptr[u];
    IDType end = csr.row_ptr[u+1];

    bool u_in_cut = (partition[u] == 2);

    for(IDType j=start; j<end; j++){
      IDType v = csr.col[j];

      bool v_in_cut = (partition[v] == 2);

      if(u_in_cut && v_in_cut){
        for(int i=0; i<np; i++){
          edges[i].emplace_back(u,v);
        }
      } else if(partition[u] == partition[v] || v_in_cut){
        if(partition[u] != 2)
          edges[partition[u]].emplace_back(u,v);
      } else if(u_in_cut){
        if(partition[v] != 2)
          edges[partition[v]].emplace_back(u,v);
      }
    }
  }


  for(int i=0; i<np; i++) {
    sort(edges[i].begin(), edges[i].end(), less<pair<IDType, IDType>>());
    auto unique_it = unique(edges[i].begin(), edges[i].end(), [](const pair<IDType,IDType>& p1, const pair<IDType,IDType>& p2){
	    return (p1.first == p2.first) && (p1.second == p2.second);
	  });
    edges[i].erase(unique_it, edges[i].end());
  }

  cout << "Constructing csrs..." << endl;
  csrs.resize(np, nullptr);
  for(int i=0; i<np; i++){
    IDType n = csr.n;
    IDType m = edges[i].size();
    IDType *row_ptr = new IDType[n+1];
    IDType *col = new IDType[m];
    IDType *csr_nodes = new IDType[n];

    fill(row_ptr, row_ptr+n+1, 0);

    row_ptr[0] = 0;

    IDType edge_index = 0;
    for(IDType j=0; j<n; j++){
      IDType count = 0;
      while(edge_index < m && edges[i][edge_index].first == j){
        auto& edge = edges[i][edge_index];
        col[edge_index] = edge.second;
        count++;
        edge_index++;
      }
      row_ptr[j+1] = count + row_ptr[j];
    }

    csrs[i] = new CSR(row_ptr, col, n, m);
  }

  int index = 0;
  for(auto csr : csrs){
    cout << "P" << index << " N:" << csr->n << endl;
    cout << "P" << index << " M:" << csr->m << endl;
    index++;
  }
}
*/

inline VertexCut::~VertexCut(){
  if(partition != nullptr)
    delete [] partition;  
}

inline VertexCut::VertexCut(CSR& csr, string part_file, string order_method, int np){

  cout << "Ordering..." << endl;
  order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  cout << "Ranking..." << endl;
  ranks.resize(csr.n);
  for(IDType i=0; i<csr.n; i++){
    ranks[order[i]] = i;
  }

  ifstream part_ifs(part_file);
 
  partition = new IDType[csr.n];
  int x;
  int i = 0;
  while(part_ifs >> x){
    partition[i++] = x;
  }

  vector<bool> in_cut(csr.n, false);
  cout << "Calculating cut..." << endl;
  for(int u=0; u<csr.n; u++){
    int start = csr.row_ptr[u];
    int end = csr.row_ptr[u+1];

    for(int j=start; j<end; j++){
      int v = csr.col[j];

      if(partition[u] != partition[v]){
        if (ranks[u] > ranks[v]){
          if(!in_cut[v]){
            cut.insert(u);
            in_cut[u] = true;
          }
        } else {
          if(!in_cut[u]){
            cut.insert(v);
            in_cut[v] = true;
          }
        }
      }
    }
  }

  cout << "Cut Size: " << cut.size() << endl;

  cout << "Calculating edges and nodes..." << endl;
  vector<vector<pair<IDType,IDType>>> edges(np);
  for(IDType u=0; u<csr.n; u++){
    IDType start = csr.row_ptr[u];
    IDType end = csr.row_ptr[u+1];

    bool u_in_cut = in_cut[u];

    for(int j=start; j<end; j++){
      IDType v = csr.col[j];

      bool v_in_cut = in_cut[v];

      if(u_in_cut && v_in_cut){
        for(int i=0; i<np; i++){
          edges[i].emplace_back(u,v);
        }
      } else if(partition[u] == partition[v] || v_in_cut){
        edges[partition[u]].emplace_back(u,v);
      } else if(u_in_cut){
        edges[partition[v]].emplace_back(u,v);
      }
    }
  }


  for(int i=0; i<np; i++) {
    sort(edges[i].begin(), edges[i].end(), less<pair<IDType, IDType>>());
    auto unique_it = unique(edges[i].begin(), edges[i].end(), [](const pair<IDType,IDType>& p1, const pair<IDType,IDType>& p2){
	    return (p1.first == p2.first) && (p1.second == p2.second);
	  });
    edges[i].erase(unique_it, edges[i].end());
  }

  cout << "Constructing csrs..." << endl;
  csrs.resize(np, nullptr);
  for(int i=0; i<np; i++){
    IDType n = csr.n;
    IDType m = edges[i].size();
    IDType *row_ptr = new IDType[n+1];
    IDType *col = new IDType[m];

    fill(row_ptr, row_ptr+n+1, 0);

    row_ptr[0] = 0;

    IDType edge_index = 0;
    for(IDType j=0; j<n; j++){
      IDType count = 0;
      while(edge_index < m && edges[i][edge_index].first == j){
        auto& edge = edges[i][edge_index];
        col[edge_index] = edge.second;
        count++;
        edge_index++;
      }
      row_ptr[j+1] = count + row_ptr[j];
    }

    csrs[i] = new CSR(row_ptr, col, n, m);
    cout << "M for P" << i << ": " << m << endl;
  }

}

#endif
