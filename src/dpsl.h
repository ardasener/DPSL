#ifndef DPSL_H
#define DPSL_H

#include "common.h"
#include "cut.h"
#include "external/toml/toml.h"
#include "mpi.h"
#include "psl.h"
#include <algorithm>
#include <fstream>
#include <ostream>
#include <string>
#include <unordered_set>
#include <set>

using namespace std;

enum MPI_CONSTS {
  MPI_GLOBAL_N,
  MPI_CSR_ROW_PTR,
  MPI_CSR_COL,
  MPI_CSR_NODES,
  MPI_CUT,
  MPI_PARTITION,
  MPI_CACHE,
  MPI_UPDATED,
  MPI_BP_DIST,
  MPI_BP_SET,
  MPI_LABEL_INDICES,
  MPI_LABELS,
  MPI_VERTEX_RANKS,
  MPI_VERTEX_ORDER,
};

class DPSL {

public:
  template <typename T>
  void SendData(T *data, int size, int vertex, int to,
                MPI_Datatype type = MPI_INT32_T);

  template <typename T>
  void BroadcastData(T *data, int size, int vertex,
                     MPI_Datatype type = MPI_INT32_T);

  template <typename T>
  int RecvData(T *&data, int vertex, int from, MPI_Datatype type = MPI_INT32_T);

  bool MergeCut(vector<vector<int> *> new_labels, PSL &psl);
  void Barrier();
  int pid, np, global_n;
  CSR *whole_csr = nullptr;
  CSR *part_csr = nullptr;
  BP *global_bp = nullptr;
  int *partition;
  vector<int> cut;
  vector<bool> in_cut;
  vector<int> names;
  vector<int> ranks;
  vector<int> order;
  VertexCut *vc_ptr = nullptr;
  int last_dist;
  const toml::Value &config;
  char **caches;
  void InitP0(string vsep_file="");
  void Init();
  void Index();
  void WriteLabelCounts(string filename);
  void Query(int u, string filename);
  void Log(string msg);
  void PrintTime(string tag, double time);
  PSL *psl_ptr;
  DPSL(int pid, CSR *csr, const toml::Value &config, int np, string vsep_file = "");
  ~DPSL();
};

inline void DPSL::Log(string msg) {
#ifdef DEBUG
  cout << "P" << pid << ": " << msg << endl;
#endif
}

inline void DPSL::PrintTime(string tag, double time) {
  cout << "P" << pid << ": " << tag << ", " << time << " seconds" << endl;
}


inline void DPSL::Query(int u, string filename) {

  Log("Starting Query");
  Log("Global N: " + to_string(global_n));
  Barrier();

  int *vertices_u;
  int *dist_ptrs_u;
  int dist_ptrs_u_size;
  int vertices_u_size;

  if (partition[u] == pid || (partition[u] == np && pid == 0)) {
    Log("Broadcasting u's labels");
    auto &labels_u = psl_ptr->labels[u];
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

  Barrier();

  char *cache = new char[part_csr->n];
  fill(cache, cache + part_csr->n, MAX_DIST);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
  for (int d = 0; d < dist_ptrs_u_size - 1; d++) {
    int start = dist_ptrs_u[d];
    int end = dist_ptrs_u[d + 1];

    for (int i = start; i < end; i++) {
      int v = vertices_u[i];
      cache[v] = d;
    }
  }

  Log("Querying locally");
  vector<int> local_dist(part_csr->n, MAX_DIST);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
  for (int v = 0; v < part_csr->n; v++) {

    int min = MAX_DIST;

    if constexpr (USE_GLOBAL_BP) {
      min = global_bp->QueryByBp(u, v);
    }

    if constexpr (USE_LOCAL_BP) {
      int local_bp_dist = psl_ptr->local_bp->QueryByBp(u, v);
      if (local_bp_dist < min) {
        min = local_bp_dist;
      }
    }

    auto &vertices_v = psl_ptr->labels[v].vertices;
    auto &dist_ptrs_v = psl_ptr->labels[v].dist_ptrs;

    for (int d = 0; d < last_dist && d < min; d++) {
      int start = dist_ptrs_v[d];
      int end = dist_ptrs_v[d + 1];

      for (int i = start; i < end; i++) {
        int w = vertices_v[i];

        int dist = d + (int) cache[w];
        if (dist < min) {
          min = dist;
        }
      }
    }

    local_dist[v] = min;
  }

  Barrier();

  Log("Synchronizing query results");
  if (pid == 0) {
    int *all_dists = new int[whole_csr->n];
    int *source = new int[whole_csr->n];
    fill(all_dists, all_dists + whole_csr->n, -1);
    fill(source, source + whole_csr->n, -1);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
    for (int i = 0; i < local_dist.size(); i++) {
      all_dists[i] = local_dist[i];
      source[i] = 0;
    }

    for (int p = 1; p < np; p++) {
      int *dists;
      int size = RecvData(dists, 0, p);
      for (int i = 0; i < size; i++) {
        if (all_dists[i] < 0 || all_dists[i] > dists[i]) {
          all_dists[i] = dists[i];
          source[i] = p;
        }
      }
      delete[] dists;
    }

    vector<int> *bfs_results = BFSQuery(*whole_csr, u);

    bool all_correct = true;
    ofstream ofs(filename);
    ofs << "Vertex\tDPSL(source)\tBFS\tCorrectness" << endl;
    for (int i = 0; i < whole_csr->n; i++) {
      int psl_res = all_dists[i];
      if (psl_res == MAX_DIST) {
        psl_res = -1;
      }
      int bfs_res = (*bfs_results)[i];
      string correctness = (bfs_res == psl_res) ? "correct" : "wrong";
      
      if (bfs_res != psl_res){
        all_correct = false;
      }
      
      ofs << i << "\t" << psl_res << "(" << source[i] << ")"
          << "\t" << bfs_res << "\t" << correctness << endl;
    }

    delete bfs_results;
    delete[] all_dists;
    delete[] source;
    ofs.close();

    cout << "Correctness of Query: " << all_correct << endl;

  } else {
    SendData(local_dist.data(), local_dist.size(), 0, 0);
  }
}

inline void DPSL::WriteLabelCounts(string filename) {

  Barrier();
  CSR &csr = *part_csr;

  int *counts = new int[part_csr->n];
  fill(counts, counts + part_csr->n, -1);

  for (int i = 0; i < part_csr->n; i++) {
    counts[i] = psl_ptr->labels[i].vertices.size();
  }

  if (pid != 0) {
    SendData(counts, part_csr->n, 0, 0);
    delete[] counts;
  }

  if (pid == 0) {

    int *source = new int[whole_csr->n];
    int *all_counts = new int[whole_csr->n];
    fill(all_counts, all_counts + whole_csr->n, -1);
    fill(source, source + whole_csr->n,
         -1); // -1 indicates free floating vertex

    for (int i = 0; i < part_csr->n; i++) {
      all_counts[i] = counts[i];
      source[i] = 0; // 0 indicates cut vertex as well as partition 0
    }

    for (int p = 1; p < np; p++) {
      int *recv_counts;
      int size = RecvData(recv_counts, 0, p);
      for (int i = 0; i < size; i++) {
        if (recv_counts[i] != -1) { // Count recieved
          if (vc_ptr->cut.find(i) == vc_ptr->cut.end() &&
              all_counts[i] < recv_counts[i]) { // vertex not in cut and counted
                                                // on the recieved data

            all_counts[i] = recv_counts[i]; // Update count
            source[i] = p;
          }
        }
      }
    }

    long long *total_per_source = new long long[np];
    fill(total_per_source, total_per_source + np, 0);
    for (int i = 0; i < part_csr->n; i++) {
      if (!in_cut[i]) {
        total_per_source[source[i]] += all_counts[i];
      } else {
        for (int j = 0; j < np; j++)
          total_per_source[j] += all_counts[i];
      }
    }

    ofstream ofs(filename);
    ofs << "Vertex\tLabelCount\tSource" << endl;
    long long total = 0;
    for (int u = 0; u < whole_csr->n; u++) {
      ofs << u << ":\t";
      ofs << all_counts[u] << "\t" << source[u];
      ofs << endl;
      total += all_counts[u];
    }
    ofs << endl;

    ofs << "Total Label Count: " << total << endl;
    cout << "Total Label Count: " << total << endl;
    ofs << "Avg. Label Count: " << total / (double)whole_csr->n << endl;
    cout << "Avg. Label Count: " << total / (double)whole_csr->n << endl;

    for (int p = 0; p < np; p++) {
      ofs << "Total for P" << p << ": " << total_per_source[p] << endl;
      ofs << "Avg for P" << p << ": "
          << total_per_source[p] / (double)whole_csr->n << endl;
    }

    ofs << endl;

    ofs.close();

    delete[] all_counts;
    delete[] source;
    delete[] total_per_source;
  }
}


inline bool DPSL::MergeCut(vector<vector<int> *> new_labels, PSL &psl) {

  bool updated = false;
  
  vector<int> compressed_labels;
  vector<int> compressed_label_indices(cut.size()+1, 0);
  // compressed_label_indices[0] = 0;
  
#pragma omp parallel default(shared) num_threads(NUM_THREADS)
{
  int tid = omp_get_thread_num();
  for(int i=tid; i<cut.size(); i+=NUM_THREADS){
    int u = cut[i];

    if(new_labels[u] != nullptr){
      int size = new_labels[u]->size();
      compressed_label_indices[i+1] = size;
      sort(new_labels[u]->begin(), new_labels[u]->end());
    } else {
      compressed_label_indices[i+1] = 0;
    }
  }
}

  
  for(int i=1; i<cut.size()+1; i++){
    compressed_label_indices[i] += compressed_label_indices[i-1];
  }

  int total_size = compressed_label_indices[cut.size()];

  if(total_size > 0){
    compressed_labels.resize(total_size, -1);
   
#pragma omp parallel default(shared) num_threads(NUM_THREADS)
    {
      int tid = omp_get_thread_num();

      for(int i=tid; i<cut.size(); i+=NUM_THREADS){
        int u = cut[i];

        int index = compressed_label_indices[i];
        
        if(new_labels[u] != nullptr)
          for(int j=0; j < new_labels[u]->size(); j++){
            compressed_labels[index++] = (*new_labels[u])[j];
          }
      }
    }
  } 

  int * compressed_merged;
  int * compressed_merged_indices;
  int compressed_size = 0;

  if(pid == 0){ // ONLY P0
    vector<int*> all_compressed_labels(np);
    vector<int*> all_compressed_label_indices(np);

    all_compressed_labels[0] = compressed_labels.data();
    all_compressed_label_indices[0] = compressed_label_indices.data();

    for(int p=1; p<np; p++){
      RecvData(all_compressed_label_indices[p], MPI_LABEL_INDICES, p);
      RecvData(all_compressed_labels[p], MPI_LABELS, p);
    }

    vector<vector<int>*> sorted_vecs(cut.size());
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) reduction(+ : compressed_size) schedule(runtime)
    for(int i=0; i<cut.size(); i++){

      vector<int>* sorted_vec = new vector<int>;
      
      for(int p=0; p<np; p++){
        

        int start = all_compressed_label_indices[p][i];
        int end = all_compressed_label_indices[p][i+1];

        if(start == end){
          continue;
        }

        int prev_size = sorted_vec->size();
        sorted_vec->insert(sorted_vec->end(), all_compressed_labels[p] + start, all_compressed_labels[p] + end);

        if(sorted_vec->size() > prev_size){
          inplace_merge(sorted_vec->begin(), sorted_vec->begin() + prev_size, sorted_vec->end());
        }
      }

      auto unique_it = unique(sorted_vec->begin(), sorted_vec->end());
      sorted_vec->erase(unique_it, sorted_vec->end());
      compressed_size += sorted_vec->size(); 
      sorted_vecs[i] = sorted_vec;
    }

    for(int p=1; p<np; p++){
      delete[] all_compressed_labels[p];
      delete[] all_compressed_label_indices[p];
    }

    compressed_merged = new int[compressed_size];
    compressed_merged_indices = new int[cut.size()+1];
    compressed_merged_indices[0] = 0;
    int index = 0;

    for(int i=0; i<cut.size(); i++){
      if(sorted_vecs[i]->size() > 0)
        copy(sorted_vecs[i]->begin(), sorted_vecs[i]->end(), compressed_merged + index);
      
      index += sorted_vecs[i]->size();
      compressed_merged_indices[i+1] = index;
      delete sorted_vecs[i];
    }


    BroadcastData(compressed_merged_indices, cut.size()+1, MPI_LABEL_INDICES);
    BroadcastData(compressed_merged, compressed_size, MPI_LABELS);
    
  } else { // ALL EXCEPT P0
    SendData(compressed_label_indices.data(), compressed_label_indices.size(), MPI_LABEL_INDICES, 0);
    SendData(compressed_labels.data(), compressed_labels.size(), MPI_LABELS, 0);
    RecvData(compressed_merged_indices, MPI_LABEL_INDICES, 0);
    compressed_size = RecvData(compressed_merged, MPI_LABELS, 0);
  }

  if(compressed_size > 0){
    updated = true;

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
    for(int i=0; i<cut.size(); i++){
      int u = cut[i];
      auto& labels_u = psl.labels[u].vertices;
      
      int start = compressed_merged_indices[i];
      int end = compressed_merged_indices[i+1];

      if(start != end){
        labels_u.insert(labels_u.end(), compressed_merged + start, compressed_merged + end);
      }

      int max = -1;
      for(int j=start; j<end; j++){
        int v = compressed_merged[j];
        if(psl.ranks[v] > max){
          max = psl.ranks[v];
        }
      }

      psl.max_ranks[u] = max;

    }

  }

  if(pid == 0){
    delete[] compressed_merged;
    delete[] compressed_merged_indices;
  }


  return updated;
}


inline void DPSL::Barrier() { MPI_Barrier(MPI_COMM_WORLD); }

template <typename T>
inline void DPSL::SendData(T *data, int size, int vertex, int to,
                           MPI_Datatype type) {
  int tag = (vertex << 1);
  int size_tag = tag | 1;

  MPI_Send(&size, 1, MPI_INT32_T, to, size_tag, MPI_COMM_WORLD);

  if (size != 0 && data != nullptr)
    MPI_Send(data, size, type, to, tag, MPI_COMM_WORLD);
}

template <typename T>
inline void DPSL::BroadcastData(T *data, int size, int vertex,
                                MPI_Datatype type) {
  for (int p = 0; p < np; p++) {
    if (p != pid)
      SendData(data, size, vertex, p, type);
  }
}

template <typename T>
inline int DPSL::RecvData(T *&data, int vertex, int from, MPI_Datatype type) {
  int tag = (vertex << 1);
  int size_tag = tag | 1;
  int size = 0;

  int error_code1, error_code2;

  error_code1 = MPI_Recv(&size, 1, MPI_INT32_T, from, size_tag, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

  if (size != 0) {
    data = new T[size];
    error_code2 = MPI_Recv(data, size, type, from, tag, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
    Log("Recieved Data with codes= " + to_string(error_code1) + "," +
        to_string(error_code2) + " and with size=" + to_string(size));
  } else {
    data = nullptr;
    Log("Recieved Size 0 Data");
  }

  return size;
}

inline void DPSL::InitP0(string vsep_file) {

  double init_start = omp_get_wtime();

  string order_method = config.find("order_method")->as<string>();
  CSR &csr = *whole_csr;
  global_n = csr.n;

  if(vsep_file == "")
    vc_ptr = new VertexCut(csr, order_method, np, config);
  else
    vc_ptr = new VertexCut(csr, vsep_file, order_method);

  VertexCut &vc = *vc_ptr;

  Log("Creating All Cut");
  cut.insert(cut.end(), vc.cut.begin(), vc.cut.end());

  in_cut.resize(global_n, false);
  for(int u : cut){
    in_cut[u] = true;
  }

  ranks = vc.ranks;
  order = vc.order;

  Log("Ordering Cut By Rank");
  sort(cut.begin(), cut.end(),
       [this](int u, int v) { return ranks[u] > ranks[v]; });

  auto &csrs = vc.csrs;
  part_csr = csrs[0];

  Log("Initial Barrier Region");
  Barrier();
  for (int i = 1; i < np; i++) {
    SendData(&global_n, 1, MPI_GLOBAL_N, i);
    SendData(csrs[i]->row_ptr, (csrs[i]->n) + 1, MPI_CSR_ROW_PTR, i);
    SendData(csrs[i]->col, csrs[i]->m, MPI_CSR_COL, i);
    SendData(vc.partition, csr.n, MPI_PARTITION, i);
    SendData(cut.data(), cut.size(), MPI_CUT, i);
    SendData(ranks.data(), ranks.size(), MPI_VERTEX_RANKS, i);
    SendData(order.data(), order.size(), MPI_VERTEX_ORDER, i);
    delete csrs[i];
    csrs[i] = nullptr;
  }

  partition = vc.partition;

  Barrier();
  Log("Initial Barrier Region End");

  if constexpr (USE_GLOBAL_BP) {
    Log("Creating Global BP");
    global_bp = new BP(csr, vc.ranks, vc.order, &cut);

    Log("Global BP Barrier Region");
    Barrier();
    for (int i = 0; i < global_n; i++) {
      BPLabel &bp_label = global_bp->bp_labels[i];
      BroadcastData(bp_label.bp_dists, N_ROOTS, MPI_BP_DIST, MPI_UINT8_T);
      vector<uint64_t> bp_sets;
      bp_sets.reserve(N_ROOTS * 2);
      for (int j = 0; j < N_ROOTS; j++) {
        bp_sets.push_back(bp_label.bp_sets[j][0]);
        bp_sets.push_back(bp_label.bp_sets[j][1]);
      }
      BroadcastData(bp_sets.data(), bp_sets.size(), MPI_BP_SET, MPI_UINT64_T);
      Barrier();
    }
    Barrier();
    Log("Global BP Barrier Region End");
  }

  Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));
  Log("Cut Size: " + to_string(cut.size()));

  double init_end = omp_get_wtime();
}

inline void DPSL::Init() {
  int *row_ptr;
  int *col;
  int *cut_ptr;
  int *ranks_ptr;
  int *order_ptr;

  Log("Initial Barrier Region");
  Barrier();

  int *global_n_ptr;
  RecvData(global_n_ptr, MPI_GLOBAL_N, 0);
  global_n = *global_n_ptr;
  delete[] global_n_ptr;

  int size_row_ptr = RecvData(row_ptr, MPI_CSR_ROW_PTR, 0);
  int size_col = RecvData(col, MPI_CSR_COL, 0);
  int size_partition = RecvData(partition, MPI_PARTITION, 0);
  int size_cut = RecvData(cut_ptr, MPI_CUT, 0);
  int size_ranks = RecvData(ranks_ptr, MPI_VERTEX_RANKS, 0);
  int size_order = RecvData(order_ptr, MPI_VERTEX_ORDER, 0);
  Barrier();
  Log("Initial Barrier Region End");

  ranks.insert(ranks.end(), ranks_ptr, ranks_ptr + size_ranks);
  order.insert(order.end(), order_ptr, order_ptr + size_order);

  if constexpr (USE_GLOBAL_BP) {
    Log("Global BP Barrier Region");
    Barrier();
    vector<BPLabel> bp_labels(global_n);
    for (int i = 0; i < global_n; i++) {
      uint8_t *bp_dists;
      RecvData(bp_dists, MPI_BP_DIST, 0, MPI_UINT8_T);
      copy(bp_dists, bp_dists + N_ROOTS, bp_labels[i].bp_dists);
      Log("Recieved dists");

      uint64_t *bp_sets;
      RecvData(bp_sets, MPI_BP_SET, 0, MPI_UINT64_T);
      Log("Recieved sets");

      for (int j = 0; j < N_ROOTS; j++) {
        bp_labels[i].bp_sets[j][0] = bp_sets[j * 2];
        bp_labels[i].bp_sets[j][1] = bp_sets[j * 2 + 1];
      }

      delete[] bp_dists;
      delete[] bp_sets;
      Barrier();
    }
    Barrier();
    Log("Global BP Barrier Region End");

    global_bp = new BP(bp_labels);
  }

  cut.insert(cut.end(), cut_ptr, cut_ptr + size_cut);

  in_cut.resize(global_n, false);
  for(int u : cut){
    in_cut[u] = true;
  }

  part_csr = new CSR(row_ptr, col, size_row_ptr - 1, size_col);
  Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));
  delete[] cut_ptr;
}

inline void DPSL::Index() {

  Barrier();
  double start, end, alg_start, alg_end;
  double total_merge_time = 0;
  Log("Indexing Start");
  CSR &csr = *part_csr;

  caches = new char *[NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; i++) {
    caches[i] = new char[part_csr->n];
    fill(caches[i], caches[i] + part_csr->n, MAX_DIST);
  }

  string order_method = config.find("order_method")->as<string>();

  vector<int>* ranks_ptr = nullptr;
  vector<int>* order_ptr = nullptr;
  if(GLOBAL_RANKS) {
    ranks_ptr = &ranks;
    order_ptr = &order;
  }
  psl_ptr = new PSL(*part_csr, order_method, &cut, global_bp, ranks_ptr, order_ptr);
  PSL &psl = *psl_ptr;

  start = omp_get_wtime();
  alg_start = omp_get_wtime();
  vector<vector<int> *> init_labels(csr.n, nullptr);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
  for (int u = 0; u < csr.n; u++) {
    psl.labels[u].vertices.push_back(u);
    init_labels[u] = psl.Init(u);
  }

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
  for (int u = 0; u < csr.n; u++) {
    if (!in_cut[u] && init_labels[u] != nullptr &&
        !init_labels[u]->empty()) {
      auto &labels = psl.labels[u].vertices;
      labels.insert(labels.end(), init_labels[u]->begin(),
                    init_labels[u]->end());
    }
  }
  end = omp_get_wtime();
  PrintTime("Level 0&1", end - start);

  Log("Merging Initial Labels");
  start = omp_get_wtime();
  MergeCut(init_labels, psl);
  end = omp_get_wtime();
  PrintTime("Merge 0&1", end - start);
  total_merge_time += end-start;
  Log("Merging Initial Labels End");

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
  for (int u = 0; u < csr.n; u++) {
    auto &labels = psl.labels[u];
    labels.dist_ptrs.push_back(0);
    labels.dist_ptrs.push_back(1);
    labels.dist_ptrs.push_back(labels.vertices.size());
    delete init_labels[u];
  }

  Barrier();

  bool should_run[csr.n];
  fill(should_run, should_run + csr.n, true);

  Log("Starting DN Loop");
  bool updated = true;
  last_dist = 1;
  for (int d = 2; d < MAX_DIST; d++) {

    Barrier();
    vector<vector<int> *> new_labels(csr.n, nullptr);

    start = omp_get_wtime();
    last_dist = d;
    updated = false;
    Log("Pulling...");
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) reduction(|| : updated) schedule(runtime)
    for (int u = 0; u < csr.n; u++) {
      /* cout << "Pulling for u=" << u << endl; */
      if (should_run[u]) {
        new_labels[u] = psl.Pull(u, d, caches[omp_get_thread_num()]);
        if (new_labels[u] != nullptr && !new_labels[u]->empty()) {
          updated = updated || true;
        }
      }
    }
    end = omp_get_wtime();
    PrintTime("Level " + to_string(d), end - start);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
    for (int u = 0; u < csr.n; u++) {
      if (new_labels[u] != nullptr && !new_labels[u]->empty()) {
        auto &labels = psl.labels[u].vertices;
        labels.insert(labels.end(), new_labels[u]->begin(),
                      new_labels[u]->end());
      }
    }

    Barrier();

    Log("Merging Labels for d=" + to_string(d));
    start = omp_get_wtime();
    bool merge_res = MergeCut(new_labels, psl);
    updated = updated || merge_res;
    end = omp_get_wtime();
    total_merge_time += end-start;
    PrintTime("Merge " + to_string(d), end - start);
    Log("Merging Labels for d=" + to_string(d) + " End");

    fill(should_run, should_run + csr.n, false);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
    for (int u = 0; u < csr.n; u++) {
      auto &labels_u = psl.labels[u];
      labels_u.dist_ptrs.push_back(labels_u.vertices.size());

      if (new_labels[u] != nullptr) {
        delete new_labels[u];
        new_labels[u] = nullptr;
      }
     
      int labels_u_dist_size = labels_u.dist_ptrs.size(); 
      if(labels_u.dist_ptrs[labels_u_dist_size-2] != labels_u.dist_ptrs[labels_u_dist_size-1]){
        int start_neighbors = csr.row_ptr[u];
        int end_neighbors = csr.row_ptr[u + 1];
        for (int i = start_neighbors; i < end_neighbors; i++) {
          int v = csr.col[i];
          if (psl.ranks[v] < psl.max_ranks[u]) {
            should_run[v] = true;
          }
        }
      }

      psl.max_ranks[u] = -1;
    }

    // Stops the execution once all processes agree that they are done
    int updated_int = (int)updated;
    BroadcastData(&updated_int, 1, MPI_UPDATED);
    for (int i = 0; i < np - 1; i++) {
      int *updated_other;
      RecvData(updated_other, MPI_UPDATED, MPI_ANY_SOURCE);
      updated_int |= *updated_other;
      delete[] updated_other;
    }

#ifdef DEBUG
    psl.CountStats(omp_get_wtime() - alg_start);
#endif

    if (updated_int == 0) {
      break;
    }
  }

  alg_end = omp_get_wtime();
  PrintTime("Total", alg_end - alg_start);
  PrintTime("Total Merge Time", total_merge_time);
  PrintTime("Total Index Time", alg_end - alg_start - total_merge_time);

#ifdef DEBUG
  Log("Prune by Rank: " + to_string(psl_ptr->prune_rank));
  Log("Prune by Local BP: " + to_string(psl_ptr->prune_local_bp));
  Log("Prune by Global BP: " + to_string(psl_ptr->prune_global_bp));
  Log("Prune by Labels: " + to_string(psl_ptr->prune_labels));
  WriteStats(psl.stats_vec, "stats_p" + to_string(pid) + ".txt");
#endif
}

inline DPSL::DPSL(int pid, CSR *csr, const toml::Value &config, int np, string vsep_file)
    : whole_csr(csr), pid(pid), config(config), np(np) {
  if (pid == 0) {
    InitP0(vsep_file);
  } else {
    Init();
  }
}

inline DPSL::~DPSL() {

  for (int i = 0; i < NUM_THREADS; i++) {
    delete[] caches[i];
  }
  delete[] caches;

  delete part_csr;

  delete psl_ptr;

  if (vc_ptr == nullptr)
    delete[] partition;
  else
    delete vc_ptr;

  if (global_bp != nullptr)
    delete global_bp;
}

#endif
