#include "cut.h"

VertexCut::~VertexCut(){
  if(partition != nullptr)
    delete [] partition;  
}

VertexCut::VertexCut(CSR& csr, string part_file, string order_method, int np){

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
  IDType i = 0;
  while(part_ifs >> x){
    partition[i++] = x;
  }

  part_ifs.close();

  cout << "Final i=" << i << endl;

  vector<bool> in_cut(csr.n, false);
  cout << "Calculating cut..." << endl;
  for(IDType i=csr.n-1; i>=0; i--){
    IDType u = order[i];
    IDType start = csr.row_ptr[u];
    IDType end = csr.row_ptr[u+1];

    for(IDType j=start; j<end; j++){
      IDType v = csr.col[j];

      if(partition[u] != partition[v]){

        if(in_cut[v] || in_cut[u]){
          continue;
        }

        if (ranks[u] > ranks[v]){
          cut.insert(u);
          in_cut[u] = true;
        } else {
          cut.insert(v);
          in_cut[v] = true;
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
