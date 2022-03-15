#ifndef DPSL_H
#define DPSL_H

#include "external/toml/toml.h"
#include "psl.h"
#include "mpi.h"
#include <algorithm>

using namespace std;

class VertexCut{
public:
  unordered_set<int> cut;
  vector<CSR*> csrs;
  int* partition;
  vector<int> ranks;
  vector<vector<int>> aliasses;

  VertexCut(CSR& csr, string order_method, int np);
};

inline VertexCut::VertexCut(CSR& csr, string order_method, int np){

  vector<int> order;
  tie(order, csr.row_ptr, csr.col, csr.n, csr.m) = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  ranks.resize(csr.n);
	for(int i=0; i<csr.n; i++){
		ranks[order[i]] = i;
	}

  int objval;
  partition = new int[csr.n];

  cout << "Partitioning..." << endl;
  int nw = 1;
  METIS_PartGraphKway(&csr.n, &nw, csr.row_ptr, csr.col,
				       NULL, NULL, NULL, &np, NULL,
				       NULL, NULL, &objval, partition);

  
  cout << "Calculating cut..." << endl;
  for(int u=0; u<csr.n; u++){
    int start = csr.row_ptr[u];
    int end = csr.col[u];

    for(int j=start; j<end; j++){
      int v = csr.col[j];

      if(partition[u] != partition[v]){
        if (ranks[u] > ranks[v]){
          cut.insert(u);
        } else {
          cut.insert(v);
        }
      }
    }
  }

  cout << "Calculating edges and nodes..." << endl;
  vector<vector<pair<int,int>>> edges(np);
  vector<unordered_set<int>> nodes(np);
  for(int u=0; u<csr.n; u++){
    int start = csr.row_ptr[u];
    int end = csr.col[u];

    for(int j=start; j<end; j++){
      int v = csr.col[j];

      bool u_in_cut = (cut.find(u) != cut.end());
      bool v_in_cut = (cut.find(v) != cut.end());

      if(partition[u] == partition[v] || v_in_cut){
        edges[partition[u]].emplace_back(u,v);
        nodes[partition[u]].insert({u,v});
      } else if(u_in_cut && v_in_cut){
        for(int i=0; i<edges.size(); i++){
          edges[i].emplace_back(u,v);
          nodes[i].insert({u,v});
        }
      } else if(u_in_cut){
        edges[partition[v]].emplace_back(u,v);
        nodes[partition[v]].insert({u,v});
      }
    }
  }

  cout << "Constructing csrs..." << endl;
  csrs.resize(np, nullptr);
  aliasses.resize(np, vector<int>(csr.n, -1));
  for(int i=0; i<np; i++){
    int n = nodes[i].size();
    int m = edges[i].size();
    int *row_ptr = new int[n+1];
    int *col = new int[m];

    fill(row_ptr, row_ptr+n+1, 0);

    sort(edges[i].begin(), edges[i].end(), less<pair<int,int>>());

    vector<int> nodes_i;
    nodes_i.insert(nodes_i.end() ,nodes[i].begin(), nodes[i].end());
    sort(nodes_i.begin(), nodes_i.end(), less<int>());

    int new_index = 0;
    for(int node: nodes_i){
      aliasses[i][node] = new_index;
    }

    for(auto& edge: edges[i]){
        edge.first = aliasses[i][edge.first];
        edge.second = aliasses[i][edge.second];
    }

    int mt = 0;
    for (auto &e : edges[i]) {
      row_ptr[e.first + 1]++;
      col[mt++] = e.second;
    }

    for (int i = 1; i <= n; i++) {
      row_ptr[i] += row_ptr[i - 1];
    }

    
    for (int i = n; i > 0; i--) {
      row_ptr[i] = row_ptr[i - 1];
    }

    row_ptr[0] = 0; 


    csrs[i] = new CSR(row_ptr, col, n, m);

  }
  
}



class DPSL{

public:
    void SendData(int* data, int size, int vertex, int to);
    void BroadcastData(int* data, int size, int vertex);
    int RecvData(int*& data,int vertex, int from);
    void MergeCut(vector<vector<int>*> new_labels, PSL& psl);
    void Barrier();
    int pid, np;
    CSR* whole_csr;
    CSR* part_csr;
    vector<int> cut;
    const toml::Value& config;
    void InitP0();
    void Init();
    void IndexP0();
    void Index();
    DPSL(int pid, CSR* csr, const toml::Value& config, int np);
};

inline void DPSL::MergeCut(vector<vector<int>*> new_labels, PSL& psl){
  if(pid == 0){
    for(int i=0; i<cut.size(); i++){
      int u = cut[i];
      auto& labels_u = psl.labels[u].vertices;
      int start = labels_u.size();
      vector<int> merged_labels;

      for(int p=1; p<np; p++){
        int* recv_labels;
        int size = RecvData(recv_labels, i, p);
        merged_labels.insert(merged_labels.end(), recv_labels, recv_labels+size);
        delete[] recv_labels;
      }

      merged_labels.insert(merged_labels.end(), new_labels[u]->begin(), new_labels[u]->end());

      labels_u.insert(labels_u.begin(), unique(merged_labels.begin(), merged_labels.end()), merged_labels.end());

      BroadcastData(labels_u.data() + start, labels_u.size()-start, i);

    }
  } else {

    for(int i=0; i<cut.size(); i++){
      int u = cut[i];
      auto& labels_u = psl.labels[u].vertices;
      SendData(new_labels[u]->data(), new_labels[u]->size(), i, 0);
      int* merged_labels;
      int size = RecvData(merged_labels, i, 0);
      
      if(size > 0){
        labels_u.insert(labels_u.end(), merged_labels, merged_labels + size);
      }
    }
  }
}

inline void DPSL::Barrier(){
    MPI_Barrier(MPI_COMM_WORLD);
}

inline void DPSL::SendData(int* data, int size, int vertex, int to){
  int tag = (vertex << 1);
  int size_tag = tag | 1;
  MPI_Send(&size, 1, MPI_INT32_T, to, size_tag, MPI_COMM_WORLD);
	MPI_Send(data, size, MPI_INT32_T, to, tag, MPI_COMM_WORLD);
}

// TODO: Replace this with MPI_Bcast
inline void DPSL::BroadcastData(int *data, int size, int vertex){
  for(int p=1; p<np; p++){
    SendData(data, size, vertex, p);
  }
}

inline int DPSL::RecvData(int *& data, int vertex, int from){
    int tag = (vertex << 1);
    int size_tag = tag | 1;
    int size = 0;
    MPI_Recv(&size, 1, MPI_INT32_T, from, size_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if(size != 0){
      data = new int[size];
      MPI_Recv(data, size, MPI_INT32_T, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      data = nullptr;
    }

    return size;
}


inline void DPSL::InitP0(){
	string order_method = config.find("order_method")->as<string>();
    
    CSR& csr = *whole_csr;

    VertexCut vc(csr, order_method, np);

    cut.insert(cut.begin(),vc.cut.begin(), vc.cut.end());
    auto& ranks = vc.ranks;

    sort(cut.begin(), cut.end(), [ranks](int u, int v){
        return ranks[u] > ranks[v];
    });

    auto& csrs = vc.csrs;
    part_csr = csrs[0];

    for(int i=1; i<np; i++){
        SendData(csrs[i]->row_ptr, (csrs[i]->n)+1, 0, i);
    }
    Barrier();

    for(int i=1; i<np; i++){
        SendData(csrs[i]->col, csrs[i]->m, 0, i);
    }
    Barrier();
    for(int i=1; i<np; i++){
      vector<int> cut_alias(cut.size());
      for(int j=0; j<cut.size(); j++){
        int u = cut[j];
        int u_alias = vc.aliasses[i][u];
        cut_alias[j] = u_alias;
      }
      SendData(cut_alias.data(), cut.size(), 0, i);
    }
    Barrier();

}

inline void DPSL::Init(){
    int *row_ptr;
    int *col;
    int *cut_ptr;

    int size_row_ptr = RecvData(row_ptr, 0, 0);
    Barrier();
    int size_col = RecvData(col, 0, 0);
    Barrier();
    int size_cut = RecvData(cut_ptr, 0, 0);
    Barrier();

    cut.insert(cut.end(), cut_ptr, cut_ptr+size_cut);

    part_csr = new CSR(row_ptr, col, size_row_ptr-1, size_col);

    delete[] cut_ptr;

}

inline void DPSL::Index(){
  CSR& csr = *part_csr;

	string order_method = config.find("order_method")->as<string>();
    PSL psl(*part_csr, order_method, &cut);

    vector<vector<int>*> init_labels(csr.n, nullptr);
    for(int u=0; u<csr.n; u++){
      init_labels[u] = psl.Init(u);
    }
    
    for(int u=0; u<csr.n; u++){
      if(psl.ranks[u] < psl.min_cut_rank && init_labels[u] != nullptr && !init_labels[u]->empty()){
        auto& labels = psl.labels[u].vertices;
        labels.insert(labels.end(), init_labels[u]->begin(), init_labels[u]->end());
      }
    }

    MergeCut(init_labels, psl);
    
    for (int u = 0; u < csr.n; u++) {
      auto& labels = psl.labels[u];
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(1);
      labels.dist_ptrs.push_back(labels.vertices.size());
      delete init_labels[u];
    }

    Barrier();

    for(int d=2; d < MAX_DIST; d++){    
        vector<vector<int>*> new_labels(csr.n, nullptr);

        for(int u=0; u<csr.n; u++){
            new_labels[u] = psl.Pull(u,d);
        }

        for(int u=0; u<csr.n; u++){
          if(psl.ranks[u] < psl.min_cut_rank && new_labels[u] != nullptr && !new_labels[u]->empty()){
            auto& labels = psl.labels[u].vertices;
            labels.insert(labels.end(), new_labels[u]->begin(), new_labels[u]->end());
          }
        }

        MergeCut(new_labels, psl);


        for(int u=0; u<csr.n; u++){
          auto& labels_u = psl.labels[u];
          labels_u.dist_ptrs.push_back(labels_u.vertices.size());
        }

        Barrier();

    }
}

inline DPSL::DPSL(int pid, CSR* csr, const toml::Value& config, int np): whole_csr(csr), pid(pid), config(config), np(np){
    if(pid == 0){
        InitP0();
    } else {
        Init();
    }
}

#endif