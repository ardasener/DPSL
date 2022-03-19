#ifndef DPSL_H
#define DPSL_H

#include "external/toml/toml.h"
#include "psl.h"
#include "mpi.h"
#include <algorithm>
#include <fstream>
#include <ostream>
#include <string>
#include <unordered_set>
#include "metis.h"

enum MPI_CONSTS{
  MPI_CSR_ROW_PTR,
  MPI_CSR_COL,
  MPI_CSR_NODES,
  MPI_CUT,
  MPI_PARTITION,
};


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

  cout << "Ordering..." << endl;
  vector<int> order;
  order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  cout << "Ranking..." << endl;
  ranks.resize(csr.n);
	for(int i=0; i<csr.n; i++){
		ranks[order[i]] = i;
	}

  int objval;
  partition = new int[csr.n];

  /*
  METIS_OPTION_OBJTYPE, METIS_OPTION_CTYPE, METIS_OPTION_IPTYPE,
  METIS_OPTION_RTYPE, METIS_OPTION_NO2HOP, METIS_OPTION_NCUTS,
  METIS_OPTION_NITER, METIS_OPTION_UFACTOR, METIS_OPTION_MINCONN,
  METIS_OPTION_CONTIG, METIS_OPTION_SEED, METIS_OPTION_NUMBERING,
  METIS_OPTION_DBGLVL
  */
  idx_t options[METIS_NOPTIONS];
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
  options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_NODE;
  options[METIS_OPTION_RTYPE] = METIS_RTYPE_GREEDY;
  options[METIS_OPTION_NO2HOP] = 1;
  options[METIS_OPTION_NCUTS] = 1;
  options[METIS_OPTION_NITER] = 10;
  options[METIS_OPTION_UFACTOR] = 30;
  options[METIS_OPTION_MINCONN] = 0;
  options[METIS_OPTION_CONTIG] = 0;
  options[METIS_OPTION_SEED] = 42;
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO;

  cout << "Partitioning..." << endl;
  int nw = 1;
  METIS_PartGraphKway(&csr.n, &nw, csr.row_ptr, csr.col,
				       NULL, NULL, NULL, &np, NULL,
				       NULL, options, &objval, partition);


  ofstream ofs("output_vertex_cut.txt");
  ofs << "__Partition__" << endl;
  for(int i=0; i<csr.n; i++){
    ofs << partition[i] << ",";
  }
  ofs << endl;

  cout << "Calculating cut..." << endl;
  for(int u=0; u<csr.n; u++){
    int start = csr.row_ptr[u];
    int end = csr.row_ptr[u+1];

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

  ofs << "__Cut__" << endl;
  for(int vertex: cut){
    ofs << vertex << ",";
  }
  ofs << endl;

  cout << "Calculating edges and nodes..." << endl;
  vector<unordered_set<pair<int,int>, pair_hash>> edge_sets(np);
  vector<unordered_set<int>> nodes(np);
  for(int u=0; u<csr.n; u++){
    int start = csr.row_ptr[u];
    int end = csr.row_ptr[u+1];

    nodes[partition[u]].insert(u);

    bool u_in_cut = (cut.find(u) != cut.end());

    for(int j=start; j<end; j++){
      int v = csr.col[j];

      bool v_in_cut = (cut.find(v) != cut.end());

      if(u_in_cut && v_in_cut){
        for(int i=0; i<edge_sets.size(); i++){
          edge_sets[i].emplace(u,v);
          edge_sets[i].emplace(v,u);
          nodes[i].insert({u,v});
        }
      } else if(partition[u] == partition[v] || v_in_cut){
        edge_sets[partition[u]].emplace(u,v);
        edge_sets[partition[u]].emplace(v,u);
        nodes[partition[u]].insert({u,v});
      } else if(u_in_cut){
        edge_sets[partition[v]].emplace(u,v);
        edge_sets[partition[v]].emplace(v,u);
        nodes[partition[v]].insert({u,v});
      } else {
        nodes[partition[u]].insert(u);
        nodes[partition[v]].insert(v);
      }
    }
  }

  vector<vector<pair<int,int>>> edges(np);

  for(int i=0; i<np; i++) {
    edges[i].insert(edges[i].end(), edge_sets[i].begin(), edge_sets[i].end());
    cout << "Edges of part " << i << endl;
    for(auto& edge : edges[i]){
      cout << "(" << edge.first << "," << edge.second << ")" << ", ";
    }
    cout << endl;
  }

  cout << "Constructing csrs..." << endl;
  csrs.resize(np, nullptr);
  aliasses.resize(np, vector<int>(csr.n, -1));
  for(int i=0; i<np; i++){
    int n = nodes[i].size();
    int m = edges[i].size();
    int *row_ptr = new int[n+1];
    int *col = new int[m];
    int *csr_nodes = new int[n];

    fill(row_ptr, row_ptr+n+1, 0);

    sort(edges[i].begin(), edges[i].end(), less<pair<int,int>>());

    vector<int> nodes_i;
    nodes_i.insert(nodes_i.end() ,nodes[i].begin(), nodes[i].end());
    sort(nodes_i.begin(), nodes_i.end(), less<int>());

    int new_index = 0;
    for(int node: nodes_i){
      aliasses[i][node] = new_index;
      csr_nodes[new_index] = node;
      new_index++;
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


    csrs[i] = new CSR(row_ptr, col, csr_nodes, n, m);

    ofs << "__P" << i << "__" << endl;
    for(int j=0; j<n; j++){
      ofs << csr_nodes[j] << ",";
    }
    ofs << endl;

  }

  ofs.close();
  
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
    int* partition;
    vector<int> cut;
    vector<int> names;
    VertexCut* vc_ptr;
    const toml::Value& config;
    void InitP0();
    void Init();
    void IndexP0();
    void Index();
    void WriteLabelCounts(string filename);
    void Query(int u, string filename);
    void Log(string msg);
    PSL * psl_ptr;
    DPSL(int pid, CSR* csr, const toml::Value& config, int np);
};

inline void DPSL::Log(string msg){
#ifdef DEBUG
  cout << "P" << pid << ": " << msg << endl;
#endif
}

inline void DPSL::Query(int u, string filename){
  
  Log("Starting Query");
  Barrier();
    
  int* vertices_u;
  int* dist_ptrs_u;
  int dist_ptrs_u_size;
  int vertices_u_size;

  if(partition[u] == pid){
    Log("Broadcasting u's labels");
    int local_u = part_csr->nodes_inv[u];
    auto& labels_u = psl_ptr->labels[u];
    BroadcastData(labels_u.vertices.data(), labels_u.vertices.size(), 0);
    BroadcastData(labels_u.dist_ptrs.data(), labels_u.dist_ptrs.size(), 1);
    vertices_u = labels_u.vertices.data();
    vertices_u_size = labels_u.vertices.size();
    dist_ptrs_u = labels_u.dist_ptrs.data();
    dist_ptrs_u_size = labels_u.dist_ptrs.size();
  } else {
    Log("Recieving u's labels");
    vertices_u_size = RecvData(vertices_u, 0, MPI_ANY_SOURCE);
    dist_ptrs_u_size = RecvData(dist_ptrs_u, 1, MPI_ANY_SOURCE);
  }

  //TODO: Remove this
  int max = *max_element(vertices_u, vertices_u+vertices_u_size);

  Log("Constructing cache");
  int cache[max+1];
  fill(cache, cache+max+1, MAX_DIST);

  for(int d=0; d<dist_ptrs_u_size-1; d++){
    int start = dist_ptrs_u[d];
    int end = dist_ptrs_u[d+1];

    for(int i= start; i<end; i++){
      int v = vertices_u[i];
      cache[v] = d;
    }
  }

  Log("Querying locally");
  vector<int> mins;
  mins.reserve(part_csr->n);
  for(int v=0; v<part_csr->n; v++){
    int min = MAX_DIST;
    auto& vertices_v = psl_ptr->labels[v].vertices;
    auto& dist_ptrs_v = psl_ptr->labels[v].dist_ptrs;
   
    for(int d=0; d<dist_ptrs_v.size()-1 && d < min; d++){
      int start = dist_ptrs_v[d];
      int end = dist_ptrs_v[d+1];

      for(int i= start; i<end; i++){
        int w = vertices_v[i];
        
        if(w < vertices_u_size){
          int dist = cache[w] + d;
          if(dist < min){
            min = dist;
          }
        }

      }
    }
    mins.push_back(min);     
  }

  Barrier();
  
  Log("Synchronizing query results");
  if(pid == 0){
    int all_dists[whole_csr->n];
    fill(all_dists, all_dists+whole_csr->n, MAX_DIST);
    for(int p=1; p<np; p++){
      int* dists;
      int size = RecvData(dists, 0, p);
      for(int i=0; i<size; i++){
        int global_id = vc_ptr->csrs[p]->nodes[i];
        if(all_dists[global_id] > dists[i]){
          all_dists[global_id] = dists[i];
        }
      }
      delete [] dists;
    }

    for(int i=0; i<mins.size(); i++){
      int global_id = part_csr->nodes[i];
      if(all_dists[global_id] > mins[i]){
        all_dists[global_id] = mins[i];
      }
    }

    vector<int>* bfs_results = BFSQuery(*whole_csr, u);

    ofstream ofs(filename);
    for(int i=0; i<whole_csr->n; i++){
      int psl_res = all_dists[i];
      int bfs_res = (*bfs_results)[i];
      string correctness = (bfs_res == psl_res) ? "correct" : "wrong";
      ofs << i << "\t" << psl_res << "\t" << bfs_res << "\t" << correctness << endl;
    }
    delete bfs_results;
    ofs.close();

  } else {
    SendData(mins.data(), mins.size(), 0, 0);
  }
}

inline void DPSL::WriteLabelCounts(string filename){
  
  Barrier();
  CSR& csr = *part_csr;
  int max_global_id = *max_element(csr.nodes, csr.nodes+csr.n);

  int counts[max_global_id+1];
  fill(counts, counts + max_global_id+1, -1);

  for(int i=0; i<part_csr->n; i++){
    int global_node_id = csr.nodes[i];
    counts[global_node_id] = psl_ptr->labels[i].vertices.size();
  }

  if(pid != 0)
    SendData(counts, max_global_id+1, 0, 0);

  
  if(pid == 0){
    
    int source[whole_csr->n]; 
    int all_counts[whole_csr->n];
    fill(all_counts, all_counts + whole_csr->n, 0);
    fill(source, source + whole_csr->n, -1); // -1 indicates free floating vertex

    for(int i=0; i<max_global_id+1; i++){
      all_counts[i] = counts[i];
      
      if(counts[i] != -1)
        source[i] = 0;  // 0 indicates cut vertex as well as partition 0
    }

    for(int p=1; p<np; p++){
      int* recv_counts;
      int size = RecvData(recv_counts, 0, p);
      for(int i=0; i<size; i++){
        if(recv_counts[i] != -1 && vc_ptr->cut.find(i) == vc_ptr->cut.end()){  // Count recieved and vertex not in cut
          all_counts[i] = recv_counts[i]; // Update count

          if(source[i] != -1)
            source[i] = -2; // -2 indicates overwrite to non-cut vertex
          else
            source[i] = p; // Process id denotes partition
        }
      }
    }

    ofstream ofs(filename);
    ofs << "Vertex\tLabelCount\tSource" << endl;
    long long total = 0;
    for(int u=0; u<whole_csr->n; u++){
      ofs << u << ":\t";
      ofs << all_counts[u] << "\t" << source[u];
      ofs << endl;
      total += all_counts[u];
    }
    ofs << endl;

    ofs << "Total Label Count: " << total << endl;
    ofs << "Avg. Label Count: " << total/(double) whole_csr->n << endl;
    
    ofs << endl;

    ofs.close();
  }
}

inline void DPSL::MergeCut(vector<vector<int>*> new_labels, PSL& psl){
  if(pid == 0){
    for(int i=0; i<cut.size(); i++){
      int u = cut[i];
      auto& labels_u = psl.labels[u].vertices;
      int start = labels_u.size();
      unordered_set<int> merged_labels;

      Log("Recieving Labels for " + to_string(i));
      for(int p=1; p<np; p++){
        int* recv_labels;
        int size = RecvData(recv_labels, i, p);
        if(size != 0 && recv_labels != nullptr){
          merged_labels.insert(recv_labels, recv_labels+size);
          delete[] recv_labels;
        }
      }

      Log("Adding Self Labels for " + to_string(i));
      if(new_labels[u] != nullptr && !new_labels[u]->empty()){
        merged_labels.insert(new_labels[u]->begin(), new_labels[u]->end());
      }
      
      Log("Merging Labels for " + to_string(i));
      labels_u.insert(labels_u.end(), merged_labels.begin(), merged_labels.end());

      Log("Broadcasting Labels for " + to_string(i));
      if(labels_u.size() > start){
        BroadcastData(labels_u.data() + start, labels_u.size()-start, i);
      } else {
        BroadcastData(nullptr, 0, i);
      }

    }
  } else {

    for(int i=0; i<cut.size(); i++){
      int u = cut[i];
      auto& labels_u = psl.labels[u].vertices;
      int new_labels_size = (new_labels[u] == nullptr) ? 0 : new_labels[u]->size();
      int* new_labels_data = (new_labels[u] == nullptr) ? nullptr : new_labels[u]->data();
      Log("Sending Labels for " + to_string(i) + " with size " + to_string(new_labels_size));
      SendData(new_labels_data, new_labels_size, i, 0);
      int* merged_labels;
      Log("Recieving Labels for " + to_string(i));
      int size = RecvData(merged_labels, i, 0);
      Log("Recieving Labels for " + to_string(i) + " End");
      
      if(size > 0){
        labels_u.insert(labels_u.end(), merged_labels, merged_labels + size);
        delete[] merged_labels;
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

  if(size != 0 && data != nullptr)
    MPI_Send(data, size, MPI_INT32_T, to, tag, MPI_COMM_WORLD);
}

// TODO: Replace this with MPI_Bcast
inline void DPSL::BroadcastData(int *data, int size, int vertex){
  for(int p=0; p<np; p++){
    if(p != pid)
      SendData(data, size, vertex, p);
  }
}

inline int DPSL::RecvData(int *& data, int vertex, int from){
    int tag = (vertex << 1);
    int size_tag = tag | 1;
    int size = 0;

    int error_code1, error_code2;

    error_code1 = MPI_Recv(&size, 1, MPI_INT32_T, from, size_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if(size != 0){
      data = new int[size];
      error_code2 = MPI_Recv(data, size, MPI_INT32_T, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      Log("Recieved Data with codes= " + to_string(error_code1) + "," + to_string(error_code2) + " and with size=" + to_string(size));
    } else {
      data = nullptr;
      Log("Recieved Size 0 Data");
    }


    return size;
}


inline void DPSL::InitP0(){
	string order_method = config.find("order_method")->as<string>();
    CSR& csr = *whole_csr;

    vc_ptr = new VertexCut(csr, order_method, np);
    VertexCut& vc = *vc_ptr;

    vector<int> all_cut(vc.cut.begin(), vc.cut.end());
    auto& ranks = vc.ranks;

    sort(all_cut.begin(), all_cut.end(), [ranks](int u, int v){
        return ranks[u] > ranks[v];
    });

    auto& csrs = vc.csrs;
    part_csr = new CSR(*csrs[0]);

    Log("Initial Barrier Region");
    Barrier();
    for(int i=1; i<np; i++){
        SendData(csrs[i]->row_ptr, (csrs[i]->n)+1, MPI_CSR_ROW_PTR, i);
        SendData(csrs[i]->col, csrs[i]->m, MPI_CSR_COL, i);
        SendData(csrs[i]->nodes, csrs[i]->n, MPI_CSR_NODES, i);
        SendData(vc.partition, csr.n, MPI_PARTITION, i);
    }

    partition = vc.partition;

    for(int i=1; i<np; i++){
      vector<int> cut_alias(all_cut.size());
      for(int j=0; j<all_cut.size(); j++){
        int u = all_cut[j];
        int u_alias = vc.aliasses[i][u];
        cut_alias[j] = u_alias;
      }
      SendData(cut_alias.data(), cut_alias.size(), MPI_CUT, i);
    }
    
    cut.resize(all_cut.size());
    for(int j=0; j<all_cut.size(); j++){
        int u = all_cut[j];
        int u_alias = vc.aliasses[0][u];
        cut[j] = u_alias;
    }

    Barrier();
    Log("Initial Barrier Region End");

    Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));
    Log("Cut Size: " + to_string(cut.size()));

    // cout << "row_ptr0:";
    // for(int i=0; i<part_csr->n+1; i++){
    //   cout << part_csr->row_ptr[i] << ",";
    // }
    // cout << endl;


    // cout << "col0:";
    // for(int i=0; i<part_csr->m; i++){
    //   cout << part_csr->col[i] << ",";
    // }
    // cout << endl;


}

inline void DPSL::Init(){
    int *row_ptr;
    int *col;
    int *cut_ptr;
    int * nodes;
    int * nodes_inv;

    Log("Initial Barrier Region");
    Barrier();
    int size_row_ptr = RecvData(row_ptr, MPI_CSR_ROW_PTR, 0);
    int size_col = RecvData(col, MPI_CSR_COL, 0);
    int size_nodes = RecvData(nodes, MPI_CSR_NODES, 0);
    int size_partition = RecvData(partition, MPI_PARTITION, 0);
    int size_cut = RecvData(cut_ptr, MPI_CUT, 0);
    Barrier();
    Log("Initial Barrier Region End");

    cut.insert(cut.end(), cut_ptr, cut_ptr+size_cut);

    part_csr = new CSR(row_ptr, col, nodes, size_row_ptr-1, size_col);
    Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));

/*     cout << "row_ptr1:"; */
/*     for(int i=0; i<part_csr->n+1; i++){ */
/*       cout << part_csr->row_ptr[i] << ","; */
/*     } */
/*     cout << endl; */


/*     cout << "col1:"; */
/*     for(int i=0; i<part_csr->m; i++){ */
/*       cout << part_csr->col[i] << ","; */
/*     } */
/*     cout << endl; */

    delete[] cut_ptr;

}

inline void DPSL::Index(){
  Log("Indexing Start");
  CSR& csr = *part_csr;

    string order_method = config.find("order_method")->as<string>();
    psl_ptr = new PSL(*part_csr, order_method, &cut);
    PSL& psl = *psl_ptr;

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

    Log("Merging Initial Labels");
    MergeCut(init_labels, psl);
    Log("Merging Initial Labels End");
    
    for (int u = 0; u < csr.n; u++) {
      auto& labels = psl.labels[u];
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(1);
      labels.dist_ptrs.push_back(labels.vertices.size());
      delete init_labels[u];
    }

    Barrier();

    Log("Starting DN Loop");
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
        
        Log("Merging Labels for d=" + to_string(d));
        MergeCut(new_labels, psl);
        Log("Merging Labels for d=" + to_string(d) + " End");


        for(int u=0; u<csr.n; u++){
          auto& labels_u = psl.labels[u];
          labels_u.dist_ptrs.push_back(labels_u.vertices.size());
          delete new_labels[u];
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
