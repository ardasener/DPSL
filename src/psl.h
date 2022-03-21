#ifndef PSL_H
#define PSL_H

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

#define N_ROOTS 16
#define MAX_BP_THREADS 6
#define USEBP false
#define NUM_THREADS 8

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
  int *nodes;
  unordered_map<int, int> nodes_inv;
  int n;
  int m;

  CSR(CSR& csr){
    row_ptr = new int[csr.n+1];
    col = new int[csr.m];
    nodes = new int[csr.n];

    for(int i=0; i<csr.n+1; i++){
      row_ptr[i] = csr.row_ptr[i];
    }

    for(int i=0; i<csr.m; i++){
      col[i] = csr.col[i];
    }

    for(int i=0; i<csr.n; i++){
      nodes[i] = csr.nodes[i];
    }
    
    for(int i=0; i<csr.n; i++){
      nodes_inv[nodes[i]] = i;
    }

    n = csr.n;
    m = csr.m;

  }

  CSR(int * row_ptr_, int *col_, int * nodes_, int n, int m): row_ptr(row_ptr_), col(col_), nodes(nodes_), n(n), m(m) {
    for(int i=0; i<n; i++){
      nodes_inv[nodes[i]] = i;
    }
  }

  CSR(string filename) {
    PigoCOO pigo_coo(filename);

    int *coo_row = pigo_coo.x();
    int *coo_col = pigo_coo.y();
    m = pigo_coo.m();
    n = pigo_coo.n();

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

    unordered_set<pair<int, int>, pair_hash> edge_set;

    for (size_t i = 0; i < m; i++) {
      edge_set.insert(pair<int, int>(coo_row[i], coo_col[i]));
      edge_set.insert(pair<int, int>(coo_col[i], coo_row[i]));
    }

    m = edge_set.size();
    cout << "N:" << n << endl;
    cout << "M:" << m << endl;

    vector<pair<int,int>> edges(edge_set.begin(), edge_set.end());

    sort(edges.begin(), edges.end(), less<pair<int, int>>());

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

    nodes = new int[n];
  
    for(int i=0; i<n; i++){
      nodes[i] = i;
      nodes_inv[nodes[i]] = i;
    }

    delete[] coo_row;
    delete[] coo_col;
  }
};

// Bit-Parallel Labels
struct BPLabel {
  uint8_t bpspt_d[N_ROOTS];
  uint64_t bpspt_s[N_ROOTS][2];
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
  BPLabel* label_bp;
  bool* usd_bp;
  vector<int> v_vs[N_ROOTS];
  int last_dist = 2;
  int min_cut_rank = 0;

  void ConstructBPLabel();
  int BPQuery(int u, int v);
  bool BPPrune(int u, int v, int d);
  bool Prune(int u, int v, int d, const vector<char> &cache);

  vector<LabelSet> labels;
  PSL(CSR &csr_, string order_method, vector<int>* cut=nullptr);
  vector<int>* Pull(int u, int d);
  vector<int>* Init(int u);
  void Index();
  void WriteLabelCounts(string filename);
  vector<int>* Query(int u);
  void Query(int u, string filename);
  void AddLabel(vector<int>* target_labels, int w);
  int GetLabel(int u, int i);
};

inline void PSL::AddLabel(vector<int>* target_labels, int w){
  target_labels->push_back(csr.nodes[w]);
}

inline int PSL::GetLabel(int u, int i){
  return csr.nodes_inv[labels[u].vertices[i]];
}

inline PSL::PSL(CSR &csr_, string order_method, vector<int>* cut) : csr(csr_),  labels(csr.n) {

  vector<int> order;
  order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  ranks.resize(csr.n);
	for(int i=0; i<csr.n; i++){
		ranks[order[i]] = i;
	}

  if(cut != nullptr && !cut->empty()){
    int high_rank = INT_MAX;
    for(int u : *cut){
      ranks[u] = high_rank--;
    }
    min_cut_rank = high_rank;
  }
  
  
  if constexpr(USEBP){
    ConstructBPLabel();
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
        continue;
      }

      if constexpr(USEBP){
        if(BPPrune(u, w, d)){
          continue;
        }
      }

      if(Prune(u, w, d, cache)){
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

      sort(new_labels[u]->begin(), new_labels[u]->end(), [this](int u, int v){
        return this->ranks[u] > this->ranks[v];
      });

      labels_u.vertices.insert(labels_u.vertices.end(), new_labels[u]->begin(), new_labels[u]->end());
      labels_u.dist_ptrs.push_back(labels_u.vertices.size());
      delete new_labels[u];
    }

    end_time = omp_get_wtime();
    cout << "Level " << d << ": " << end_time-start_time << " seconds" << endl;
  }

  all_end_time = omp_get_wtime();
  cout << "Indexing: " << all_end_time-all_start_time << " seconds" << endl;
}

inline int PSL::BPQuery(int u, int v) {
  BPLabel &idx_u = label_bp[u], &idx_v = label_bp[v];
  int d = MAX_DIST;
  for (int i = 0; i < N_ROOTS; ++i) {
    int td = idx_u.bpspt_d[i] + idx_v.bpspt_d[i];
    if (td - 2 <= d)
      td += (idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2
            : ((idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) |
               (idx_u.bpspt_s[i][1] & idx_v.bpspt_s[i][0]))
                ? -1
                : 0;
    if (td < d)
      d = td;
  }
  return d;
}

inline bool PSL::BPPrune(int u, int v, int d) {
  BPLabel &idx_u = label_bp[u], &idx_v = label_bp[v];
  for (int i = 0; i < N_ROOTS; ++i) {
    int td = idx_u.bpspt_d[i] + idx_v.bpspt_d[i];
    if (td - 2 <= d)
      td += (idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2
            : ((idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) |
               (idx_u.bpspt_s[i][1] & idx_v.bpspt_s[i][0]))
                ? -1
                : 0;
    if (td <= d)
      return true;
  }
  return false;
}

inline void PSL::ConstructBPLabel() {
  int nown = csr.n;
  int n = csr.n;
  int m = csr.m;

  printf("Constructing BP Label...\n");
  double tt = omp_get_wtime();
  label_bp = new BPLabel[nown];
  usd_bp = new bool[n];
  memset(usd_bp, 0, sizeof(bool) * n);
  vector<int> v_vs[N_ROOTS];

  int r = 0;
  for (int i_bpspt = 0; i_bpspt < N_ROOTS; ++i_bpspt) {
    while (r < nown && usd_bp[r])
      ++r;
    if (r == nown) {
      for (int v = 0; v < nown; ++v)
        label_bp[v].bpspt_d[i_bpspt] = MAX_DIST;
      continue;
    }
    usd_bp[r] = true;
    v_vs[i_bpspt].push_back(r);
    int ns = 0;

    int start = csr.row_ptr[r];
    int end = csr.row_ptr[r];
    for (int i = start; i < end; ++i) {
      int v = csr.col[i];
      if (!usd_bp[v]) {
        usd_bp[v] = true;
        v_vs[i_bpspt].push_back(v);
        if (++ns == 64)
          break;
      }
    }
  }

  int n_threads = 1;
#pragma omp parallel
  {
    if (omp_get_thread_num() == 0)
      n_threads = omp_get_num_threads();
  }
  if (n_threads > MAX_BP_THREADS)
    omp_set_num_threads(MAX_BP_THREADS);
#pragma omp parallel
  {
    int pid = omp_get_thread_num(), np = omp_get_num_threads();
    if (pid == 0)
      printf("n_threads_bp = %d\n", np);
    vector<uint8_t> tmp_d(nown);
    vector<pair<uint64_t, uint64_t>> tmp_s(nown);
    vector<int> que(nown);
    // vector<pair<int, int> > sibling_es(m/2);
    vector<pair<int, int>> child_es(m / 2);

    for (int i_bpspt = pid; i_bpspt < N_ROOTS; i_bpspt += np) {
      printf("[%d]", i_bpspt);

      if (v_vs[i_bpspt].size() == 0)
        continue;
      fill(tmp_d.begin(), tmp_d.end(), MAX_DIST);
      fill(tmp_s.begin(), tmp_s.end(), make_pair(0, 0));

      r = v_vs[i_bpspt][0];
      int que_t0 = 0, que_t1 = 0, que_h = 0;
      que[que_h++] = r;
      tmp_d[r] = 0;
      que_t1 = que_h;

      for (size_t i = 1; i < v_vs[i_bpspt].size(); ++i) {
        int v = v_vs[i_bpspt][i];
        que[que_h++] = v;
        tmp_d[v] = 1;
        tmp_s[v].first = 1ULL << (i - 1);
      }

      for (int d = 0; que_t0 < que_h; ++d) {
        // int num_sibling_es = 0;
        int num_child_es = 0;

        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
          int v = que[que_i];
  
          int start = csr.row_ptr[v];
          int end = csr.row_ptr[v+1];
          for (int i = start; i < end; ++i) {
            int tv = csr.col[i];
            int td = d + 1;

            if (d == tmp_d[tv]) {
              if (v < tv) {
                // sibling_es[num_sibling_es].first  = v;
                // sibling_es[num_sibling_es].second = tv;
                //++num_sibling_es;
                tmp_s[v].second |= tmp_s[tv].first;
                tmp_s[tv].second |= tmp_s[v].first;
              }
            } else if (d < tmp_d[tv]) {
              if (tmp_d[tv] == MAX_DIST) {
                que[que_h++] = tv;
                tmp_d[tv] = td;
              }
              child_es[num_child_es].first = v;
              child_es[num_child_es].second = tv;
              ++num_child_es;
              // tmp_s[tv].first  |= tmp_s[v].first;
              // tmp_s[tv].second |= tmp_s[v].second;
            }
          }
        }

        /*for (int i = 0; i < num_sibling_es; ++i) {
                int v = sibling_es[i].first, w = sibling_es[i].second;
                tmp_s[v].second |= tmp_s[w].first;
                tmp_s[w].second |= tmp_s[v].first;
        }*/

        for (int i = 0; i < num_child_es; ++i) {
          int v = child_es[i].first, c = child_es[i].second;
          tmp_s[c].first |= tmp_s[v].first;
          tmp_s[c].second |= tmp_s[v].second;
        }

        que_t0 = que_t1;
        que_t1 = que_h;
      }

      for (int v = 0; v < nown; ++v) {
        label_bp[v].bpspt_d[i_bpspt] = tmp_d[v];
        label_bp[v].bpspt_s[i_bpspt][0] = tmp_s[v].first;
        label_bp[v].bpspt_s[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first;
      }
    }
  }
  omp_set_num_threads(n_threads);
  printf("\nBP Label Constructed, bp_size=%0.3lfMB, time = %0.3lf sec\n",
         sizeof(BPLabel) * nown / (1024.0 * 1024.0), omp_get_wtime() - tt);
}

#endif
