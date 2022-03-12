#pragma once

#include "external/toml/toml.h"
#include "psl.h"
#include "mpi.h"


class VertexCut{
public:
  unordered_set<int> cut;
  vector<CSR*> csrs;
  int* partition;
  vector<int> ranks;
  vector<unordered_map<int, int>> aliasses;

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
  METIS_PartGraphKway(&csr.n, &csr.m, csr.row_ptr, csr.col,
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
  aliasses.resize(np, unordered_map<int, int>());
  for(int i=0; i<np; i++){
    int n = nodes[i].size();
    int m = edges[i].size();
    int *row_ptr = new int[n+1];
    int *col = new int[m];

    fill(row_ptr, row_ptr+n+1, 0);

    sort(edges[i].begin(), edges[i].end(), less<pair<int,int>>());
    sort(nodes[i].begin(), nodes[i].end());

    int new_index = 0;
    for(int node: nodes[i]){
      aliasses[i][node] = new_index;
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
private:
    void SendData(int* data, int size, int vertex, int to);
    int RecvData(int*& data,int vertex, int from);
    void Barrier();
    int pid;
    CSR& whole_csr;
    CSR* part_csr;
    vector<int> cut;
    const toml::Value& config;
    void InitP0();
    void Init();
    void IndexP0();
    void Index();
public:
    DPSL(int pid, CSR& csr, const toml::Value& config);
};

inline void DPSL::Barrier(){
    MPI_Barrier(MPI_COMM_WORLD);
}

inline void DPSL::SendData(int* data, int size, int vertex, int to){
  int tag = (vertex << 1);
  int size_tag = tag | 1;
  MPI_Send(&size, 1, MPI_INT32_T, to, size_tag, MPI_COMM_WORLD);
	MPI_Send(data, size, MPI_INT32_T, to, tag, MPI_COMM_WORLD);
}

inline int DPSL::RecvData(int *& data, int vertex, int from){
    int tag = (vertex << 1);
    int size_tag = tag | 1;
    int size = 0;
    MPI_Recv(&size, 1, MPI_INT32_T, from, size_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    data = new int[size];
    MPI_Recv(data, size, MPI_INT32_T, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return size;
}


inline void DPSL::InitP0(){
	string order_method = config.find("order_method")->as<string>();
    int np = config.find("num_processes")->as<int>();
    
    CSR& csr = whole_csr;

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
        SendData(cut.data(), cut.size(), 0, i);
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

}

inline void DPSL::Index(){
    CSR& csr = *part_csr;

	string order_method = config.find("order_method")->as<string>();
    PSL psl(*part_csr, order_method, &cut);

    psl.Init();
    for(int i=0; i<cut.size(); i++){
        int u = cut[i];
        auto& labels = psl.labels[u].vertices;
        SendData(labels.data(), labels.size(), i, 0);
    }
    Barrier();

    for(int d=2; d < MAX_DIST; d++){    
        vector<vector<int>*> new_labels(csr.n, nullptr);

        for(int u=0; u<csr.n; u++){
            new_labels[u] = psl.Pull(u,d);
        }

        for(int i=0; i<cut.size(); i++){
            int u = cut[i];
            SendData(new_labels[], int size, int vertex, int to)
        }
    }
}

inline DPSL::DPSL(int pid, CSR& csr, const toml::Value& config): whole_csr(csr), pid(pid), config(config){
    if(pid == 0){
        InitP0();
    } else {
        Init();
    }
}