#ifndef PSL_H
#define PSL_H

#include "common.h"
#include "external/order/order.hpp"
#include "external/pigo/pigo.hpp"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>
#include <unordered_set>
#include <unordered_map>
#include "bp.h"


using namespace std;



// Stores the labels for each vertex
struct LabelSet {
  vector<int> vertices; // global vertex ids in order of distances - log(n) 
  vector<int> dist_ptrs; // indices for the vertices vector denoting distance starts - max_dist
};

enum PruneIDs {
  PRUNE_RANK,
  PRUNE_GLOBAL_BP,
  PRUNE_LOCAL_BP,
  PRUNE_LABEL,
};


class PSL {

public:
  
  CSR &csr;
  vector<int> ranks;
  vector<int> order;
  BP* local_bp;
  BP* global_bp;
  bool* usd_bp;
  vector<int> v_vs[N_ROOTS];
  int last_dist = 2;
  int min_cut_rank = 0;
  long long prune_rank = 0;
  long long prune_labels = 0;
  long long prune_local_bp = 0;
  long long prune_global_bp = 0;

  void ConstructBPLabel();

  vector<LabelSet> labels;
  PSL(CSR &csr_, string order_method, vector<int>* cut=nullptr, BP* global_bp=nullptr);
  vector<int>* Pull(int u, int d);
  vector<int>* Init(int u);
  void Index();
  void WriteLabelCounts(string filename);
  vector<int>* Query(int u);
  void AddLabel(vector<int>* target_labels, int w);
  int GetLabel(int u, int i);
  void CountPrune(int i);
  bool Prune(int u, int v, int d, const vector<char> &cache);
  void Query(int u, string filename);

};


inline void PSL::CountPrune(int i){
#ifdef DEBUG
  if(i == PRUNE_RANK)
    prune_rank++;
  else if (i == PRUNE_LOCAL_BP)
    prune_local_bp++;
  else if (i == PRUNE_GLOBAL_BP)
    prune_global_bp++;
  else
    prune_labels++;

#endif
}

inline void PSL::AddLabel(vector<int>* target_labels, int w){
  target_labels->push_back(csr.nodes[w]);
}

inline int PSL::GetLabel(int u, int i){
  return csr.nodes_inv[labels[u].vertices[i]];
}

inline PSL::PSL(CSR &csr_, string order_method, vector<int>* cut, BP* global_bp) : csr(csr_),  labels(csr.n), global_bp(global_bp) {

  order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  ranks.resize(csr.n);
	for(int i=0; i<csr.n; i++){
		ranks[order[i]] = i;
	}

  min_cut_rank = INT_MAX;

  if(cut != nullptr && !cut->empty()){
    int high_rank = INT_MAX;
    for(int u : *cut){
      ranks[u] = high_rank--;
    }
    min_cut_rank = high_rank;
  }
  
  
  if constexpr(USE_LOCAL_BP){
    local_bp = new BP(csr_, ranks, order, cut, LOCAL_BP_MODE);
  }
}


inline void PSL::WriteLabelCounts(string filename){
  ofstream ofs(filename);
  ofs << "L:\t";
  for(int i=0; i<last_dist; i++){
    ofs << i << "\t";
  }
  ofs << endl;

  long long total = 0;
  for(int u=0; u<csr.n; u++){
    ofs << u << ":\t";
    auto& labels_u = labels[u];
    total += labels_u.vertices.size();

    for(int d=0; d<last_dist; d++){
      int dist_start = labels_u.dist_ptrs[d];
      int dist_end = labels_u.dist_ptrs[d+1];

      ofs << dist_end - dist_start << "\t";
    }
    ofs << endl;
  }
  ofs << endl;

  ofs << "Total Label Count: " << total << endl;
  ofs << "Avg. Label Count: " << total/(double) csr.n << endl;
  
  ofs << endl;

  ofs.close();
}

inline void PSL::Query(int u, string filename){
  auto results = Query(u);
  auto bfs_results = BFSQuery(csr, u);
  
  ofstream ofs(filename);

  ofs << "Source: " << u << endl;
  ofs << "Target\tPSL_Distance\tBFS_Distance\tCorrectness" << endl;
  for(int i=0; i<csr.n; i++){
    int psl_res = results->at(i);
    int bfs_res = bfs_results->at(i);
    string correctness = (bfs_res == psl_res) ? "correct" : "wrong";
    ofs << i << "\t" << psl_res << "\t" << bfs_res << "\t" << correctness << endl;
  }

  ofs.close();
  delete results;
  delete bfs_results;
}

inline vector<int>* PSL::Query(int u) {


  vector<int>* results = new vector<int>(csr.n, MAX_DIST);

  auto &labels_u = labels[u];

  vector<char> cache(csr.n, -1);

  

  for (int i = 0; i < last_dist; i++) {
    int dist_start = labels_u.dist_ptrs[i];
    int dist_end = labels_u.dist_ptrs[i + 1];

    for (int d = dist_start; d < dist_end; d++) {
      int w = GetLabel(u, d); 
      cache[w] = (char) i;
    }
  }

  for (int v = 0; v < csr.n; v++) {

    auto& labels_v = labels[v];

    int min = MAX_DIST;

    if constexpr(USE_LOCAL_BP)
      min = local_bp->QueryByBp(u,v);

    for (int d = 0; d < min && d < last_dist; d++) {
      int dist_start = labels_v.dist_ptrs[d];
      int dist_end = labels_v.dist_ptrs[d + 1];

      for (int i = dist_start; i < dist_end; i++) {
        int w = GetLabel(v, i);
        
        if(cache[w] == -1){
          continue;
        }

        int dist = d + (int) cache[w];
        if(dist < min){
          min = dist;
        }
      }
    }
    
    (*results)[v] = (min == MAX_DIST) ? -1 : min;

  }
  
  return results;
}

inline bool PSL::Prune(int u, int v, int d, const vector<char> &cache) {

  auto &labels_v = labels[v];

  for (int i = 0; i < d; i++) {
    int dist_start = labels_v.dist_ptrs[i];
    int dist_end = labels_v.dist_ptrs[i + 1];

    for (int j = dist_start; j < dist_end; j++) {
      int w = GetLabel(v, j); 

      if (cache[w] != -1 && (i + cache[w]) <= d) {
        return true;
      }
    }
  }

  return false;
}

inline vector<int>* PSL::Pull(int u, int d) {

  int global_u = csr.nodes[u];
  
  int start = csr.row_ptr[u];
  int end = csr.row_ptr[u + 1];
  
  if(end-start == 0){
    return nullptr;
  }

  auto &labels_u = labels[u];
  
  vector<char> cache(csr.n, -1);
  for (int i = 0; i < d; i++) {
    int dist_start = labels_u.dist_ptrs[i];
    int dist_end = labels_u.dist_ptrs[i + 1];

    for (int j = dist_start; j < dist_end; j++) {
      int w = GetLabel(u, j);
      cache[w] = (char) i;
    }
  }

  vector<int>* new_labels = new vector<int>;
  bool updated = false;

  for (int i = start; i < end; i++) {
    int v = csr.col[i];
    auto &labels_v = labels[v];

    int labels_start = labels_v.dist_ptrs[d-1];
    int labels_end = labels_v.dist_ptrs[d];
    

    for (int j = labels_start; j < labels_end; j++) {
      int w = GetLabel(v, j);

      if (ranks[u] > ranks[w]) {
	CountPrune(PRUNE_RANK);
        continue;
      }

      if constexpr(USE_GLOBAL_BP){
        int global_w = labels_v.vertices[j];
        if(global_bp != nullptr && global_bp->PruneByBp(global_u, global_w, d)){
          CountPrune(PRUNE_GLOBAL_BP);
          continue;
        }
      }

      if constexpr(USE_LOCAL_BP){
        if(ranks[u] < min_cut_rank && ranks[w] < min_cut_rank && local_bp->PruneByBp(u, w, d)){
	  CountPrune(PRUNE_LOCAL_BP);
          continue;
        }
      }

      if(Prune(u, w, d, cache)){
	CountPrune(PRUNE_LABEL);
        continue;
      }

      AddLabel(new_labels, w);
      cache[w] = d;
    }
  }

  return new_labels;
}

inline vector<int>* PSL::Init(int u){
  
  vector<int>* init_labels = new vector<int>;

  AddLabel(init_labels, u);

  int start = csr.row_ptr[u];
  int end = csr.row_ptr[u + 1];

  for (int j = start; j < end; j++) {
    int v = csr.col[j];

    if (ranks[v] > ranks[u]) {
      AddLabel(init_labels, v);
    }
  }

  return init_labels;

}


inline void PSL::Index() {

  double start_time, end_time, all_start_time, all_end_time;
  all_start_time = omp_get_wtime();

  // Adds the first two level of vertices
  // Level 0: vertex to itself
  // Level 1: vertex to neighbors
  start_time = omp_get_wtime();
  #pragma omp parallel for default(shared) num_threads(NUM_THREADS)
  for (int u = 0; u < csr.n; u++) {
    auto init_labels = Init(u);
    labels[u].vertices.insert(labels[u].vertices.end(), init_labels->begin(), init_labels->end());
    delete init_labels;
    labels[u].dist_ptrs.push_back(0);
    labels[u].dist_ptrs.push_back(1);
    labels[u].dist_ptrs.push_back(labels[u].vertices.size());
  }
  end_time = omp_get_wtime();
  cout << "Level 0 & 1: " << end_time-start_time << " seconds" << endl;

  // bool should_run[csr.n];
  // fill(should_run, should_run+csr.n, true);

  bool updated = true;
  for (int d = 2; d < MAX_DIST && updated; d++) {
    
    start_time = omp_get_wtime();
    updated = false;

    vector<vector<int>*> new_labels(csr.n, nullptr);
    #pragma omp parallel for default(shared) num_threads(NUM_THREADS) reduction(||:updated)
    for (int u = 0; u < csr.n; u++) {

      new_labels[u] =  Pull(u, d);
      updated = updated || (new_labels[u] != nullptr && !new_labels[u]->empty());
    }

    last_dist++;

    // fill(should_run, should_run+csr.n, false);
    
    #pragma omp parallel for default(shared) num_threads(NUM_THREADS) 
    for (int u = 0; u < csr.n; u++) {
      
      auto& labels_u = labels[u];
      
      if(new_labels[u] == nullptr){
        labels_u.dist_ptrs.push_back(labels_u.vertices.size());
        continue;
      }

      /* sort(new_labels[u]->begin(), new_labels[u]->end(), [this](int u, int v){ */
      /*   return this->ranks[u] > this->ranks[v]; */
      /* }); */

      labels_u.vertices.insert(labels_u.vertices.end(), new_labels[u]->begin(), new_labels[u]->end());
      labels_u.dist_ptrs.push_back(labels_u.vertices.size());
      delete new_labels[u];
    }

    end_time = omp_get_wtime();
    cout << "Level " << d << ": " << end_time-start_time << " seconds" << endl;
  }

  all_end_time = omp_get_wtime();
  cout << "Indexing: " << all_end_time-all_start_time << " seconds" << endl;

#ifdef DEBUG
  cout << "Prune by Rank: " << prune_rank << endl; 
  cout << "Prune by Local BP: " << prune_local_bp << endl; 
  cout << "Prune by Global BP: " << prune_global_bp << endl; 
  cout << "Prune by Labels: " << prune_labels << endl; 
#endif
}

#endif
