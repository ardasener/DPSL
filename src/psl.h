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


using namespace std;


// Bit-Parallel Labels
struct BPLabel {
  uint8_t bp_dists[N_ROOTS];
  uint64_t bp_sets[N_ROOTS][2];
};

// Stores the labels for each vertex
struct LabelSet {
  vector<int> vertices; // global vertex ids in order of distances - log(n) 
  vector<int> dist_ptrs; // indices for the vertices vector denoting distance starts - max_dist
  /* char bp_dist[N_ROOTS]; */
  /* uint64_t bp_sets[N_ROOTS][2]; */
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


class PSL {

public:
  
  CSR &csr;
  vector<int> ranks;
  vector<int> order;
  BPLabel* label_bp;
  bool* usd_bp;
  vector<int> v_vs[N_ROOTS];
  int last_dist = 2;
  int min_cut_rank = 0;
  long long prune_rank = 0;
  long long prune_labels = 0;
  long long prune_bp = 0;

  void ConstructBPLabel();

  vector<LabelSet> labels;
  vector<BPLabel> bp_labels;
  PSL(CSR &csr_, string order_method, vector<int>* cut=nullptr);
  vector<int>* Pull(int u, int d);
  vector<int>* Init(int u);
  void Index();
  void WriteLabelCounts(string filename);
  vector<int>* Query(int u);
  void AddLabel(vector<int>* target_labels, int w);
  int GetLabel(int u, int i);
  void CountPrune(int i);
  void InitBP();
  void InitBPForRoot(int r, vector<int>& Sr, int root_index);
  bool PruneByBp(int u, int v, int d);
  bool Prune(int u, int v, int d, const vector<char> &cache);
  void Query(int u, string filename);
  int QueryByBp(int u, int v);
};

inline bool PSL::PruneByBp(int u, int v, int d){

  BPLabel &idx_u = bp_labels[u], &idx_v = bp_labels[v];

  for (int i = 0; i < N_ROOTS; ++i) {

      int td = idx_u.bp_dists[i] + idx_v.bp_dists[i];
      if (td - 2 <= d)
	td += (idx_u.bp_sets[i][0] & idx_v.bp_sets[i][0]) ? -2
	      : ((idx_u.bp_sets[i][0] & idx_v.bp_sets[i][1]) |
				 (idx_u.bp_sets[i][1] & idx_v.bp_sets[i][0]))
		  ? -1
		  : 0;
  
      if (td <= d){
	/* printf("BPPruned u=%d v=%d td=%d d=%d \n", u,v,td,d); */
	return true;
      }
    }
    return false;

}


inline int PSL::QueryByBp(int u, int v){

  BPLabel &idx_u = bp_labels[u], &idx_v = bp_labels[v];

  int d = MAX_DIST;
  for (int i = 0; i < N_ROOTS; ++i) {

      int td = idx_u.bp_dists[i] + idx_v.bp_dists[i];
      if (td - 2 <= d)
	td += (idx_u.bp_sets[i][0] & idx_v.bp_sets[i][0]) ? -2
	      : ((idx_u.bp_sets[i][0] & idx_v.bp_sets[i][1]) |
				 (idx_u.bp_sets[i][1] & idx_v.bp_sets[i][0]))
		  ? -1
		  : 0;
  
      if (td < d){
	d = td;
      }
    }

    return d;
}


inline void PSL::InitBPForRoot(int r, vector<int>& Sr, int bp_index){

  cout << "BP Init for root=" << r << endl;
  vector<pair<uint64_t,uint64_t>> bp_sets(csr.n, make_pair((uint64_t) 0, (uint64_t) 0));
  vector<uint8_t> bp_dists(csr.n, (uint8_t) MAX_DIST);

  int* Q0 = new int[csr.n];
  int* Q1 = new int[csr.n];
  int Q0_size = 0;
  int Q0_ptr = 0;
  int Q1_size = 0;
  int Q1_ptr = 0;

  Q0[Q0_size++] = r;
  bp_dists[r] = (uint8_t) 0;

  for(int i=0; i<Sr.size(); i++){
    int v = Sr[i];
    Q1[Q1_size++] = v;
    bp_dists[v] = (uint8_t) 1;
    bp_sets[v].first |= 1ULL << i;
  }

  while(Q0_ptr < Q0_size){
    unordered_set<pair<int,int>, pair_hash> E0;
    unordered_set<pair<int,int>, pair_hash> E1;

    while(Q0_ptr < Q0_size){
      int v = Q0[Q0_ptr++];

      int start = csr.row_ptr[v];
      int end = csr.row_ptr[v+1];

      for(int i=start; i<end; i++){
	int u = csr.col[i];

	if(bp_dists[u] == (uint8_t) MAX_DIST){
	  E1.emplace(v,u);
	  bp_dists[u] = (uint8_t) (bp_dists[v] + 1);
	  Q1[Q1_size++] = u;
	} else if(bp_dists[u] == bp_dists[v] + 1){
	  E1.emplace(v,u);
	} else if (bp_dists[u] == bp_dists[v]){
	  E0.emplace(v,u);
	} 
      } 
    }

    for(auto& p : E0){
      int v = p.first;
      int u = p.second;
      bp_sets[v].second |= bp_sets[u].first;
      bp_sets[u].second |= bp_sets[v].first;
    }

    for(auto& p : E1){
      int v = p.first;
      int u = p.second;
      bp_sets[u].first |= bp_sets[v].first;
      bp_sets[u].second |= bp_sets[v].second & ~bp_sets[v].first;
      /* bp_sets[u].second |= bp_sets[v].second; */
    }

    swap(Q0, Q1);
    swap(Q0_ptr, Q1_ptr);
    swap(Q0_size, Q1_size);
    Q1_ptr = 0;
    Q1_size = 0;
  }


  for(int i=0; i<csr.n; i++){
    auto& bp = bp_labels[i];
    bp.bp_dists[bp_index] = bp_dists[i];
    bp.bp_sets[bp_index][0] = bp_sets[i].first;
    bp.bp_sets[bp_index][1] = bp_sets[i].second;
  }

  delete[] Q0;
  delete[] Q1;

}

inline void PSL::InitBP(){
 
  cout << "Creating BP Labels" << endl; 
  bp_labels.resize(csr.n);
  for(int i=0; i<csr.n; i++){
    for(int j=0; j<N_ROOTS; j++){
      bp_labels[i].bp_dists[j] = (uint8_t) MAX_DIST;
      bp_labels[i].bp_sets[j][0] = (uint64_t) 0;
      bp_labels[i].bp_sets[j][1] = (uint64_t) 0;   
    }
  }


  vector<bool> used(csr.n, false);
  vector<int> roots;

  cout << "Sorting the neighbors" << endl;
  for(int i=0; i<csr.n; i++){
    int start = csr.row_ptr[i];
    int end = csr.row_ptr[i+1];

    sort(csr.col+start, csr.col+end, [this](int u, int v){
	return this->ranks[u] > this->ranks[v];
      });
  }


  int root_index = csr.n-1;
  for(int i=0; i<N_ROOTS; i++){

    cout << "Choosing root " << i << endl;
    vector<int> SR;
    SR.reserve(64);

    while(root_index >= 0){
      int root = order[root_index]; 
      if(!used[root]){
	break;
      }
      root_index--;
    }

    if(root_index == 0){
      cout << "Run out of root nodes for BP" << endl;
      break;
    }

    int root = order[root_index];
    roots.push_back(root);
    cout << "Chosen root=" << root << endl;
    cout << "Chosen root rank=" << ranks[root] << endl;
    used[root] = true;

    int start = csr.row_ptr[root];
    int end = csr.row_ptr[root+1];

    cout << "Computing SR" << endl;
    for(int j=start; j<end && j-start < 64; j++){
      int v = csr.col[j];
      SR.push_back(v);
      used[v] = true;
    }

    InitBPForRoot(root, SR, i);    
  }

#ifdef DEBUG
  ofstream ofs("output_bp_labels.txt");
  ofs << "__Roots__" << endl;
  for(int i=0; i<roots.size(); i++){
    ofs << roots[i] << ", ";	
  }
  ofs << endl;
  ofs << "__Dists__" << endl;
  for(int v=0; v<csr.n; v++){
	  ofs << v << ": ";
	  for(int i=0; i<N_ROOTS; i++){
		  ofs << (int) bp_labels[v].bp_dists[i] << ", ";	
	  }
	  ofs << endl;
  }
  
  ofs << "__Sets__" << endl;
  for(int v=0; v<csr.n; v++){
	  ofs << v << ": ";
	  for(int i=0; i<N_ROOTS; i++){
		  ofs << "(" << bp_labels[v].bp_sets[i][0] << "," << bp_labels[v].bp_sets[i][1] << ")" << ", ";	
	  }
	  ofs << endl;
  }

  ofs.close();




#endif

}

inline void PSL::CountPrune(int i){
#ifdef DEBUG
  if(i == 0)
    prune_rank++;
  else if (i == 1)
    prune_bp++;
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

inline PSL::PSL(CSR &csr_, string order_method, vector<int>* cut) : csr(csr_),  labels(csr.n) {

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
  
  
  if constexpr(USEBP){
    InitBP();
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
    int min = QueryByBp(u,v);

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
	CountPrune(0);
        continue;
      }

      if constexpr(USEBP){
        if(ranks[u] < min_cut_rank && ranks[w] < min_cut_rank && PruneByBp(u, w, d)){
	  CountPrune(1);
          continue;
        }
      }

      if(Prune(u, w, d, cache)){
	CountPrune(2);
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
  cout << "Prune by BP: " << prune_bp << endl; 
  cout << "Prune by Labels: " << prune_labels << endl; 
#endif
}

#endif
