#ifndef CUT_H
#define CUT_H

#include "common.h"
#include "external/order/order.hpp"
#include "external/metis/metis.h"
#include "external/kahypar/libkahypar.h"
#include "external/toml/toml.h"
/* #include "patoh_wrap.h" */
#include <vector>
#include "omp.h"
#include <cmath>

using namespace std;

void KahyparPart(CSR& csr, int*& partition, int np, string config_file, double imbalance, int* vertex_weights){

  kahypar_context_t* context = kahypar_context_new();
  kahypar_configure_context_from_file(context, config_file.c_str());

  const kahypar_hypernode_id_t num_vertices = csr.n;
  const kahypar_hyperedge_id_t num_hyperedges = csr.m;

  std::unique_ptr<size_t[]> hyperedge_indices = std::make_unique<size_t[]>(csr.m+1);
  std::unique_ptr<kahypar_hyperedge_id_t[]> hyperedges = std::make_unique<kahypar_hyperedge_id_t[]>(csr.m*2);

  int index=0;
  for(int u=0; u<csr.n; u++){
    int start = csr.row_ptr[u];
    int end = csr.row_ptr[u+1];

    for(int j=start; j<end; j++){
      int v = csr.col[j];
      hyperedges[index++] = u; 
      hyperedges[index++] = v; 
    }
  }

  for(int i=0; i<csr.m+1; i++){
    hyperedge_indices[i] = 2*i;
  }

  const kahypar_partition_id_t k = np;
  kahypar_hyperedge_weight_t objective = 0;

  partition = new int[csr.n];

  kahypar_partition(num_vertices, num_hyperedges,
                  imbalance, k,
                  vertex_weights, /*edge_weights*/ nullptr,
                  hyperedge_indices.get(), hyperedges.get(),
                  &objective, context, partition);

  kahypar_context_free(context);

}

/*
void PatohPart(CSR& csr, int*& partition, int np, string mode, string minimize, double imbalance){
  partition = new int[csr.n];

  int* ptrs = csr.row_ptr;
  int* js = csr.col;
  int m = csr.n;
  int n = csr.n;
  int nz = csr.m;
  int* partv = partition;

  if(mode == "default"){
    patoh_speed = DEFAULT;
  } else if(mode == "speed"){
    patoh_speed = SPEED;
  } else if(mode == "quality"){
    patoh_speed = QUALITY;
  } else {
    throw "Unsupported mode";
  }

  if(minimize == "cut"){
    patoh_metric = CUT;
  } else if(minimize == "con"){
    patoh_metric = CON;
  } else {
    throw "Unsupported minimization";
  }

  patoh_no_parts = np;
  patoh_imbal = imbalance;

  rowNetPart(ptrs, js, m, n, partv);
  
}
*/

void MetisPart(CSR& csr, int*& partition, int np, int* vertex_weights){

  int objval;
  partition = new int[csr.n];

  idx_t options[METIS_NOPTIONS];
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // Try others
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
				       vertex_weights, NULL, NULL, &np, NULL,
				       NULL, options, &objval, partition);


}

class VertexCut{
public:
  unordered_set<int> cut;
  vector<CSR*> csrs;
  int* partition;
  vector<int> ranks;
  vector<int> order;

  VertexCut(CSR& csr, string order_method, int np, const toml::Value& config);
  ~VertexCut();
};

inline VertexCut::~VertexCut(){
  delete partition;  
}

inline VertexCut::VertexCut(CSR& csr, string order_method, int np, const toml::Value& config){

  ofstream ofs("output_vertex_cut.txt");
  
  cout << "Ordering..." << endl;
  order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  cout << "Ranking..." << endl;
  ranks.resize(csr.n);
  for(int i=0; i<csr.n; i++){
    ranks[order[i]] = i;
  }

  int vertex_weights[csr.n];
  string partition_weight = config.find("partition_weight")->as<string>();
  if(partition_weight == "uniform"){
    fill(vertex_weights, vertex_weights + csr.n, 1);
  } else if(partition_weight == "rank"){
    for(int i=0; i<csr.n; i++){
      vertex_weights[i] = ranks[i]; 
    }
  } else if(partition_weight == "degree"){
    for(int i=0; i<csr.n; i++){
      vertex_weights[i] = csr.row_ptr[i+1] - csr.row_ptr[i];
    }
  } else if(partition_weight == "degree_log"){
    for(int i=0; i<csr.n; i++){
      int degree = csr.row_ptr[i+1] - csr.row_ptr[i];
      int degree_log = (int) log2(degree);
      vertex_weights[i] = (degree_log >= 0) ? degree_log : 0;
    }
  }

  cout << "Partitioning..." << endl;
  double start_part = omp_get_wtime();
  string partition_engine = config.find("partition_engine")->as<string>();
  if(partition_engine == "metis")
    MetisPart(csr, partition, np, vertex_weights);
  else if (partition_engine == "patoh")
    /*
    PatohPart(csr, partition, np, 
        config.find("patoh.mode")->as<string>(),
        config.find("patoh.minimize")->as<string>(),
        config.find("patoh.imbalance")->as<double>());
        */
    throw "Patoh is not supported at the moment";
  else if (partition_engine == "kahypar")
    KahyparPart(csr, partition, np, 
        config.find("kahypar.config_file")->as<string>(),
        config.find("kahypar.imbalance")->as<double>(),
        vertex_weights);
  else
   throw "Unsupported partition engine";

  double end_part = omp_get_wtime();
  ofs << "Time: " << end_part-start_part << " seconds" << endl;


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

  ofs << "Cut Size: " << cut.size() << endl;

#ifdef DEBUG
  ofs << "__Vertex Weights__" << endl;
  for(int i=0; i<csr.n; i++){
    ofs << vertex_weights[i] << ",";
  }
  ofs << endl;


  ofs << "__Partition__" << endl;
  for(int i=0; i<csr.n; i++){
    ofs << partition[i] << ",";
  }
  ofs << endl;

  ofs << "__Cut__" << endl;
  for(int vertex: cut){
    ofs << vertex << ",";
  }
  ofs << endl;
#endif


  cout << "Calculating edges and nodes..." << endl;
  vector<unordered_set<pair<int,int>, pair_hash>> edge_sets(np);
  for(int u=0; u<csr.n; u++){
    int start = csr.row_ptr[u];
    int end = csr.row_ptr[u+1];

    bool u_in_cut = (cut.find(u) != cut.end());

    for(int j=start; j<end; j++){
      int v = csr.col[j];

      bool v_in_cut = (cut.find(v) != cut.end());

      if(u_in_cut && v_in_cut){
        for(int i=0; i<edge_sets.size(); i++){
          edge_sets[i].emplace(u,v);
          edge_sets[i].emplace(v,u);
        }
      } else if(partition[u] == partition[v] || v_in_cut){
        edge_sets[partition[u]].emplace(u,v);
        edge_sets[partition[u]].emplace(v,u);
      } else if(u_in_cut){
        edge_sets[partition[v]].emplace(u,v);
        edge_sets[partition[v]].emplace(v,u);
      }
    }
  }

  vector<vector<pair<int,int>>> edges(np);

  for(int i=0; i<np; i++) {
    edges[i].insert(edges[i].end(), edge_sets[i].begin(), edge_sets[i].end());
  }

  cout << "Constructing csrs..." << endl;
  csrs.resize(np, nullptr);
  for(int i=0; i<np; i++){
    int n = csr.n;
    int m = edges[i].size();
    int *row_ptr = new int[n+1];
    int *col = new int[m];
    int *csr_nodes = new int[n];

    fill(row_ptr, row_ptr+n+1, 0);

    sort(edges[i].begin(), edges[i].end(), less<pair<int,int>>());

    row_ptr[0] = 0;

    int edge_index = 0;
    for(int j=0; j<n; j++){
      int count = 0;
      while(edge_index < m && edges[i][edge_index].first == j){
        auto& edge = edges[i][edge_index];
        col[edge_index] = edge.second;
        count++;
        edge_index++;
      }
      row_ptr[j+1] = count + row_ptr[j];
    }

    csrs[i] = new CSR(row_ptr, col, n, m);

#ifdef DEBUG
    ofs << "__Row Ptr P" << i << "__" << endl;
    for(int i=0; i<n+1; i++){
      ofs << row_ptr[i] << ",";
    }
    ofs << endl;

    ofs << "__Col P" << i << "__" << endl;
    for(int i=0; i<m; i++){
      ofs << col[i] << ",";
    }
    ofs << endl;
#endif
  }

  int index = 0;
  for(auto csr : csrs){
    ofs << "P" << index << " N:" << csr->n << endl;
    ofs << "P" << index << " M:" << csr->m << endl;
    index++;
  }

  cout << "Closing OFS" << endl;
  ofs.close();
  cout << "Closed OFS" << endl;

}

#endif
