#include "dpsl.h"

void DPSL::Log(string msg) {
#ifdef DEBUG
  cout << "P" << pid << ": " << msg << endl;
#endif
}

void DPSL::PrintTime(string tag, double time) {
  cout << "P" << pid << ": " << tag << ", " << time << " seconds" << endl;
}

void DPSL::QueryTest(int query_count){
  IDType* sources;
  if(pid == 0){
    sources = new IDType[query_count];

    size_t cut_select = random_range(0, cut.size());
    sources[0] = cut[cut_select];
    for(int i=1; i<query_count; i++){
      sources[i] = random_range(0, whole_csr->n);
    }

    BroadcastData(sources, query_count);
  } else {
    RecvBroadcast(sources, 0);
  }


  for(int i=0; i<query_count; i++){
    Barrier();
    IDType u = sources[i];
    if(pid == 0)
      cout << "Query From: " << u << "(" << partition[u] << ")" << endl;
    Query(u, "output_dpsl_query_" + to_string(i) + ".txt");
  }

  delete [] sources;
}

void DPSL::Query(IDType u, string filename) {

  Log("Starting Query");
  Log("Global N: " + to_string(global_n));
  Barrier();

  IDType *vertices_u;
  IDType *dist_ptrs_u;
  IDType dist_ptrs_u_size;
  IDType vertices_u_size;

  double start_time, end_time;

  start_time = omp_get_wtime();

  if (partition[u] == pid) {
    Log("Broadcasting u's labels");
    auto &labels_u = psl_ptr->labels[u];
    BroadcastData(labels_u.vertices.data(), labels_u.vertices.size());
    BroadcastData(labels_u.dist_ptrs.data(), labels_u.dist_ptrs.size());
    vertices_u = labels_u.vertices.data();
    vertices_u_size = labels_u.vertices.size();
    dist_ptrs_u = labels_u.dist_ptrs.data();
    dist_ptrs_u_size = labels_u.dist_ptrs.size();
  } else {
    Log("Recieving u's labels");
    vertices_u_size = RecvBroadcast(vertices_u, partition[u]);
    dist_ptrs_u_size = RecvBroadcast(dist_ptrs_u, partition[u]);
  }

  Barrier();

  char *cache = new char[part_csr->n];
  fill(cache, cache + part_csr->n, MAX_DIST);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
  for (IDType d = 0; d < dist_ptrs_u_size - 1; d++) {
    IDType start = dist_ptrs_u[d];
    IDType end = dist_ptrs_u[d + 1];

    for (IDType i = start; i < end; i++) {
      IDType v = vertices_u[i];
      cache[v] = d;
    }
  }

  Log("Querying locally");
  vector<int> local_dist(part_csr->n, MAX_DIST);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
  for (IDType v = 0; v < part_csr->n; v++) {

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
      IDType start = dist_ptrs_v[d];
      IDType end = dist_ptrs_v[d + 1];

      for (IDType i = start; i < end; i++) {
        IDType w = vertices_v[i];

        int dist = d + (int) cache[w];
        if (dist < min) {
          min = dist;
        }
      }
    }

    local_dist[v] = min;
  }

  delete[] cache;

  if(partition[u] != pid){
    delete[] vertices_u;
    delete[] dist_ptrs_u;
  }

  Barrier();

  Log("Synchronizing query results");
  if (pid == 0) {
    int *all_dists = new int[whole_csr->n];
    int *source = new int[whole_csr->n];
    fill(all_dists, all_dists + whole_csr->n, -1);
    fill(source, source + whole_csr->n, 0);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
    for (IDType i = 0; i < local_dist.size(); i++) {
      all_dists[i] = local_dist[i];
    }

    for (int p = 1; p < np; p++) {
      int *dists;
      int size = RecvData(dists, 0, p, MPI_INT32_T);
      #pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
      for (int i = 0; i < whole_csr->n; i++) {
        if(dists[i] < all_dists[i]){
          source[i] = p;
          all_dists[i] = dists[i];
        }
      }
      delete[] dists;
    }

    end_time = omp_get_wtime();

    cout << "Avg. Query Time: " << (end_time-start_time) / whole_csr->n << " seconds" << endl;

    start_time = omp_get_wtime();
    vector<int> *bfs_results = BFSQuery(*whole_csr, u);
    end_time = omp_get_wtime();

    cout << "Avg. BFS Time: " << (end_time-start_time) / whole_csr->n << " seconds" << endl;;

    bool all_correct = true;
    
#ifdef DEBUG
    ofstream ofs(filename);
    ofs << "Vertex\tDPSL(source)\tBFS\tCorrectness" << endl;
#endif
    
    for (IDType i = 0; i < whole_csr->n; i++) {
      int psl_res = all_dists[i];
      if (psl_res == MAX_DIST) {
        psl_res = -1;
      }
      int bfs_res = (*bfs_results)[i];
      string correctness = (bfs_res == psl_res) ? "correct" : "wrong";
      
      if (bfs_res != psl_res){
        all_correct = false;
      }
      
#ifdef DEBUG
      ofs << i << "\t" << psl_res << "(" << source[i] << ")"
          << "\t" << bfs_res << "\t" << correctness << endl;
#endif

    }

    delete bfs_results;
    delete[] all_dists;
    delete[] source;

#ifdef DEBUG
    ofs.close();
#endif

    cout << "Correctness of Query: " << all_correct << endl;

  } else {
    SendData(local_dist.data(), local_dist.size(), 0, 0, MPI_INT32_T);
  }
}

void DPSL::WriteLabelCounts(string filename) {

  Barrier();
  CSR &csr = *part_csr;

  IDType *counts = new IDType[part_csr->n];
  fill(counts, counts + part_csr->n, -1);

  for (IDType i = 0; i < part_csr->n; i++) {
    counts[i] = psl_ptr->labels[i].vertices.size();
  }

  if (pid != 0) {
    SendData(counts, part_csr->n, 0, 0);
    delete[] counts;
  }

  if (pid == 0) {

    int *source = new int[whole_csr->n];
    IDType *all_counts = new IDType[whole_csr->n];
    fill(all_counts, all_counts + whole_csr->n, -1);
    fill(source, source + whole_csr->n,
         -1); // -1 indicates free floating vertex

    for (IDType i = 0; i < part_csr->n; i++) {
      all_counts[i] = counts[i];
      source[i] = 0; // 0 indicates cut vertex as well as partition 0
    }

    for (int p = 1; p < np; p++) {
      IDType *recv_counts;
      IDType size = RecvData(recv_counts, 0, p);
      for (IDType i = 0; i < size; i++) {
        if (recv_counts[i] != -1) { // Count recieved
          if (vc_ptr->cut.find(i) == vc_ptr->cut.end() &&
              all_counts[i] < recv_counts[i]) { // vertex not in cut and counted
                                                // on the recieved data

            all_counts[i] = recv_counts[i]; // Update count
            source[i] = p;
          }
        }
      }
      delete[] recv_counts;
    }

    long long *total_per_source = new long long[np];
    fill(total_per_source, total_per_source + np, 0);
    for (IDType i = 0; i < part_csr->n; i++) {
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
    for (IDType u = 0; u < whole_csr->n; u++) {
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


bool DPSL::MergeCut(vector<vector<IDType> *> new_labels, PSL &psl) {

  bool updated = false;
  
  vector<IDType> compressed_labels;
  vector<IDType> compressed_label_indices(cut.size()+1, 0);
  // compressed_label_indices[0] = 0;
  

#pragma omp parallel default(shared) num_threads(NUM_THREADS)
{
  int tid = omp_get_thread_num();
  for(IDType i=tid; i<cut.size(); i+=NUM_THREADS){
    IDType u = cut[i];

    if(new_labels[u] != nullptr){
      IDType size = new_labels[u]->size();
      compressed_label_indices[i+1] = size;
      sort(new_labels[u]->begin(), new_labels[u]->end());
    } else {
      compressed_label_indices[i+1] = 0;
    }
  }
}

  
  for(IDType i=1; i<cut.size()+1; i++){
    compressed_label_indices[i] += compressed_label_indices[i-1];
  }

  IDType total_size = compressed_label_indices[cut.size()];

  if(total_size > 0){
    compressed_labels.resize(total_size, -1);
   
#pragma omp parallel default(shared) num_threads(NUM_THREADS)
    {
      int tid = omp_get_thread_num();

      for(IDType i=tid; i<cut.size(); i+=NUM_THREADS){
        IDType u = cut[i];

        IDType index = compressed_label_indices[i];
        
        if(new_labels[u] != nullptr)
          for(IDType j=0; j < new_labels[u]->size(); j++){
            compressed_labels[index++] = (*new_labels[u])[j];
          }
      }
    }
  } 

  IDType * compressed_merged;
  IDType * compressed_merged_indices;
  IDType compressed_size = 0;

  if(pid == 0){ // ONLY P0
    vector<IDType*> all_compressed_labels(np);
    vector<IDType*> all_compressed_label_indices(np);

    all_compressed_labels[0] = compressed_labels.data();
    all_compressed_label_indices[0] = compressed_label_indices.data();

    for(int p=1; p<np; p++){
      RecvData(all_compressed_label_indices[p], MPI_LABEL_INDICES, p);
      RecvData(all_compressed_labels[p], MPI_LABELS, p);
    }

    vector<vector<IDType>*> sorted_vecs(cut.size());
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) reduction(+ : compressed_size) schedule(SCHEDULE)
    for(IDType i=0; i<cut.size(); i++){

      vector<IDType>* sorted_vec = new vector<IDType>;
      //TODO: Reserve
      
      for(IDType p=0; p<np; p++){
        

        IDType start = all_compressed_label_indices[p][i];
        IDType end = all_compressed_label_indices[p][i+1];

        if(start == end){
          continue;
        }

        IDType prev_size = sorted_vec->size();
        sorted_vec->insert(sorted_vec->end(), all_compressed_labels[p] + start, all_compressed_labels[p] + end);

        inplace_merge(sorted_vec->begin(), sorted_vec->begin() + prev_size, sorted_vec->end());
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

    compressed_merged = new IDType[compressed_size];
    compressed_merged_indices = new IDType[cut.size()+1];
    compressed_merged_indices[0] = 0;
    IDType index = 0;

#pragma omp parallel default(shared) num_threads(NUM_THREADS)
{
    int tid = omp_get_thread_num();
    for(IDType i=tid; i<cut.size(); i+=NUM_THREADS){
      compressed_merged_indices[i+1] = sorted_vecs[i]->size();
    }
}


    for(IDType i=1; i<cut.size()+1; i++){
      compressed_merged_indices[i] += compressed_merged_indices[i-1];
    }



#pragma omp parallel default(shared) num_threads(NUM_THREADS)
{
    int tid = omp_get_thread_num();
    for(IDType i=tid; i<cut.size(); i+=NUM_THREADS){
      if(sorted_vecs[i]->size() > 0)
        copy(sorted_vecs[i]->begin(), sorted_vecs[i]->end(), compressed_merged + compressed_merged_indices[i]);
      delete sorted_vecs[i];
      sorted_vecs[i] = nullptr;
    }
}

    BroadcastData(compressed_merged_indices, cut.size()+1);
    BroadcastData(compressed_merged, compressed_size);
    
  } else { // ALL EXCEPT P0
    SendData(compressed_label_indices.data(), compressed_label_indices.size(), MPI_LABEL_INDICES, 0);
    SendData(compressed_labels.data(), compressed_labels.size(), MPI_LABELS, 0);
    RecvBroadcast(compressed_merged_indices, 0);
    compressed_size = RecvBroadcast(compressed_merged, 0);
  }

  if(compressed_size > 0){
    updated = true;

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
    for(IDType i=0; i<cut.size(); i++){
      IDType u = cut[i];
      auto& labels_u = psl.labels[u].vertices;
      
      IDType start = compressed_merged_indices[i];
      IDType end = compressed_merged_indices[i+1];

      if(start != end){
        labels_u.insert(labels_u.end(), compressed_merged + start, compressed_merged + end);
      }

      IDType max = -1;
      for(IDType j=start; j<end; j++){
        IDType v = compressed_merged[j];
        if(v > max){
          max = v;
        }
      }

      psl.max_ranks[u] = max;

    }

  }

  delete[] compressed_merged;
  delete[] compressed_merged_indices;


  return updated;
}


void DPSL::Barrier() { MPI_Barrier(MPI_COMM_WORLD); }

template <typename T>
void DPSL::SendData(T *data, int size, int tag, int to,
                           MPI_Datatype type) {
  int data_tag = (tag << 1);
  int size_tag = data_tag | 1;

  MPI_Send(&size, 1, MPI_INT32_T, to, size_tag, MPI_COMM_WORLD);

  if (size != 0 && data != nullptr)
    MPI_Send(data, size, type, to, data_tag, MPI_COMM_WORLD);
}

template <typename T>
void DPSL::BroadcastData(T *data, int size, MPI_Datatype type) {

  Barrier();
  // cout << "Broadcasting size..." << endl;
  MPI_Bcast(&size, 1, MPI_INT32_T, pid, MPI_COMM_WORLD);
  Barrier();
  if(size != 0 && data != nullptr)
    // cout << "Broadcasting data... size=" << size << endl;
    MPI_Bcast(data, size, type, pid, MPI_COMM_WORLD);
}

template <typename T>
int DPSL::RecvBroadcast(T *&data, int from, MPI_Datatype type){
  int size = 0;
  Barrier();
  // cout << "Recv size broadcast..." << endl;
  MPI_Bcast(&size, 1, MPI_INT32_T, from, MPI_COMM_WORLD);
  Barrier();
  if(size != 0){
    data = new T[size];
    // cout << "Recv data broadcast... size=" << size << endl;
    MPI_Bcast(data, size, type, from, MPI_COMM_WORLD); 
  } else {
    data = nullptr;
  }
  return size;  
}

template <typename T>
int DPSL::RecvData(T *&data, int tag, int from, MPI_Datatype type) {
  int data_tag = (tag << 1);
  int size_tag = data_tag | 1;
  int size = 0;

  int error_code1, error_code2;

  error_code1 = MPI_Recv(&size, 1, MPI_INT32_T, from, size_tag, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

  if (size != 0) {
    data = new T[size];
    error_code2 = MPI_Recv(data, size, type, from, data_tag, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
    Log("Recieved Data with codes= " + to_string(error_code1) + "," +
        to_string(error_code2) + " and with size=" + to_string(size));
  } else {
    data = nullptr;
    Log("Recieved Size 0 Data");
  }

  return size;
}

void DPSL::InitP0(string part_file) {

  double init_start = omp_get_wtime();

  string order_method = ORDER_METHOD;
  CSR &csr = *whole_csr;
  global_n = csr.n;

  if(part_file == "")
    throw "Partition file is required";
  else
    vc_ptr = new VertexCut(csr, part_file, order_method, np);

  VertexCut &vc = *vc_ptr;

  Log("Creating All Cut");
  cut.insert(cut.end(), vc.cut.begin(), vc.cut.end());

  in_cut.resize(global_n, false);
  for(IDType u : cut){
    in_cut[u] = true;
  }

  ranks = vc.ranks;
  order = vc.order;

  Log("Ordering Cut By Rank");
  sort(cut.begin(), cut.end(),
       [this](IDType u, IDType v) { return ranks[u] < ranks[v]; });

  auto &csrs = vc.csrs;

  csr.Reorder(order, &cut, &in_cut);
  
  for(int p=0; p<np; p++)
    csrs[p]->Reorder(order, &cut, &in_cut);
  
  part_csr = csrs[0];

  ofstream cut_ofs("output_dpsl_cut.txt");
  for(int i=0; i<cut.size(); i++){
    cut_ofs << cut[i] << endl;
  }
  cut_ofs.close();

// #pragma omp parallel default(shared) num_threads(NUM_THREADS)
  for(int i=0; i < cut.size(); i++){
    cut[i] = csr.reorder_ids[cut[i]];
  }

  

  in_cut.clear();
  in_cut.resize(global_n, false);
  for(IDType u : cut){
    in_cut[u] = true;
  }


  partition = new IDType[csr.n]; 
  for(int i=0; i<csr.n; i++){
    IDType new_id = csr.reorder_ids[i];
    partition[new_id] = vc.partition[i];
  }

  // unordered_set<IDType> cut_set;
  // cut_set.insert(cut.begin(), cut.end());
  // cout << "Cut is unique: " << (cut_set.size() == cut.size()) << endl;
  // cout << "Cut: " << cut.size() << endl;
  // cout << "Cut Set: " << cut_set.size() << endl;


  double comm_start = omp_get_wtime();

  Log("Initial Barrier Region");
  Barrier();
  for (int i = 1; i < np; i++) {
    SendData(&global_n, 1, MPI_GLOBAL_N, i);
    SendData(csrs[i]->row_ptr, (csrs[i]->n) + 1, MPI_CSR_ROW_PTR, i);
    SendData(csrs[i]->col, csrs[i]->m, MPI_CSR_COL, i);
    SendData(partition, csr.n, MPI_PARTITION, i);
    SendData(cut.data(), cut.size(), MPI_CUT, i);
    SendData(ranks.data(), ranks.size(), MPI_VERTEX_RANKS, i);
    SendData(order.data(), order.size(), MPI_VERTEX_ORDER, i);
    delete csrs[i];
    csrs[i] = nullptr;
  }

  Barrier();
  Log("Initial Barrier Region End");

  if constexpr (USE_GLOBAL_BP) {
    Log("Creating Global BP");
    global_bp = new BP(csr, &cut);

    Log("Global BP Barrier Region");
    Barrier();

    BroadcastData(global_bp->bp_dists.data(), global_bp->bp_dists.size(), MPI_UINT8_T);
    BroadcastData(global_bp->bp_0_sets.data(), global_bp->bp_0_sets.size(), MPI_UINT64_T);
    BroadcastData(global_bp->bp_1_sets.data(), global_bp->bp_1_sets.size(), MPI_UINT64_T);

    int* bp_used = new int[global_n];

#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for(int i=0; i<global_n; i++){
      bp_used[i] = (int) global_bp->used[i];
    }
    BroadcastData(bp_used, global_n);
    delete[] bp_used;

    Barrier();
    Log("Global BP Barrier Region End");
  }

  Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));
  Log("Cut Size: " + to_string(cut.size()));

  double init_end = omp_get_wtime();
  double comm_end = omp_get_wtime();

  cout << "Init Total: " << init_end - init_start << " seconds" << endl; 
  cout << "Init Communication: " << comm_end - comm_start << " seconds" << endl; 
}

void DPSL::Init() {
  IDType *row_ptr;
  IDType *col;
  IDType *cut_ptr;
  IDType *ranks_ptr;
  IDType *order_ptr;

  Log("Initial Barrier Region");
  Barrier();

  IDType *global_n_ptr;
  RecvData(global_n_ptr, MPI_GLOBAL_N, 0);
  global_n = *global_n_ptr;
  delete[] global_n_ptr;

  IDType size_row_ptr = RecvData(row_ptr, MPI_CSR_ROW_PTR, 0);
  IDType size_col = RecvData(col, MPI_CSR_COL, 0);
  IDType size_partition = RecvData(partition, MPI_PARTITION, 0);
  IDType size_cut = RecvData(cut_ptr, MPI_CUT, 0);
  IDType size_ranks = RecvData(ranks_ptr, MPI_VERTEX_RANKS, 0);
  IDType size_order = RecvData(order_ptr, MPI_VERTEX_ORDER, 0);
  Barrier();
  Log("Initial Barrier Region End");

  ranks.insert(ranks.end(), ranks_ptr, ranks_ptr + size_ranks);
  order.insert(order.end(), order_ptr, order_ptr + size_order);

  if constexpr (USE_GLOBAL_BP) {
    Log("Global BP Barrier Region");
    Barrier();

    uint8_t *bp_dists;
    size_t bp_dists_size = RecvBroadcast(bp_dists, 0, MPI_UINT8_T);
    uint64_t *bp_0_sets;
    size_t bp_0_sets_size = RecvBroadcast(bp_0_sets, 0, MPI_UINT64_T);
    uint64_t *bp_1_sets;
    size_t bp_1_sets_size = RecvBroadcast(bp_1_sets, 0, MPI_UINT64_T);

    vector<uint64_t> bp_0_sets_vec(bp_0_sets, bp_0_sets + bp_0_sets_size);
    vector<uint64_t> bp_1_sets_vec(bp_1_sets, bp_1_sets + bp_1_sets_size);
    vector<uint8_t> bp_dists_vec(bp_dists, bp_dists + bp_dists_size);

    delete[] bp_dists;
    delete[] bp_0_sets;
    delete[] bp_1_sets;

    int* bp_used; 
    size_t bp_used_size = RecvBroadcast(bp_used, 0);

    vector<bool> bp_used_vec(global_n);

    #pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for(int i=0; i<global_n; i++){
      bp_used_vec[i] = (bool) bp_used[i];
    } 

    delete[] bp_used;
    Barrier();
    Log("Global BP Barrier Region End");

    global_bp = new BP(bp_0_sets_vec, bp_1_sets_vec, bp_dists_vec, bp_used_vec);
  }

  cut.insert(cut.end(), cut_ptr, cut_ptr + size_cut);

  in_cut.resize(global_n, false);
  for(IDType u : cut){
    in_cut[u] = true;
  }

  part_csr = new CSR(row_ptr, col, size_row_ptr - 1, size_col);
  Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));
  delete[] cut_ptr;
}

void DPSL::Index() {

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

  used = new vector<bool>[NUM_THREADS];
  #pragma omp parallel for default(shared) num_threads(NUM_THREADS)
  for(int i=0; i<NUM_THREADS; i++){
    used[i].resize(csr.n, false);
  }

  bool should_run[csr.n];
  fill(should_run, should_run + csr.n, true);

  string order_method = ORDER_METHOD;

  vector<IDType>* ranks_ptr = &ranks;
  vector<IDType>* order_ptr = &order;

  psl_ptr = new PSL(*part_csr, order_method, &cut, global_bp, ranks_ptr, order_ptr);
  PSL &psl = *psl_ptr;

  start = omp_get_wtime();
  alg_start = omp_get_wtime();
  vector<vector<IDType> *> init_labels(csr.n, nullptr);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
  for (IDType u = 0; u < csr.n; u++) {

    bool should_init = true; 
    if constexpr(USE_GLOBAL_BP){
      if(global_bp->used[u])
        should_init = false;
    }

    if(should_init){
      psl.labels[u].vertices.push_back(u);
      init_labels[u] = psl.Init(u);
    }
  }

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
  for (IDType u = 0; u < csr.n; u++) {
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

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
  for (IDType u = 0; u < csr.n; u++) {
    auto &labels = psl.labels[u];

    bool should_init = true; 
    if constexpr(USE_GLOBAL_BP){
      if(global_bp->used[u])
        should_init = false;
    }

    if(should_init){
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(1);
      labels.dist_ptrs.push_back(labels.vertices.size());

      IDType ngh_start = csr.row_ptr[u];
      IDType ngh_end = csr.row_ptr[u+1];

      for(IDType i=ngh_start; i<ngh_end; i++){
        IDType v = csr.col[i];

        if(v < psl.max_ranks[u])
          should_run[v] = true;
      }

      psl.max_ranks[u] = -1;

      delete init_labels[u];
    } else {
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(0);
    }

  }

  Barrier();


  IDType* nodes_to_process = new IDType[csr.n];
  IDType num_nodes = 0;
  for(IDType u=0; u<csr.n; u++){
    if(csr.row_ptr[u] != csr.row_ptr[u+1]){
      nodes_to_process[num_nodes++] = u;
    }
  }

  Log("Starting DN Loop");
  bool updated = true;
  last_dist = 1;
  for (int d = 2; d < MAX_DIST; d++) {

    Barrier();
    vector<vector<IDType> *> new_labels(csr.n, nullptr);

    start = omp_get_wtime();
    last_dist = d;
    updated = false;
    Log("Pulling...");
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) reduction(|| : updated) schedule(SCHEDULE)
    for (IDType i = 0; i < num_nodes; i++) {

      int tid = omp_get_thread_num();

      IDType u = nodes_to_process[i];
      /* cout << "Pulling for u=" << u << endl; */

      if (should_run[u]) {

        if constexpr(SMART_DIST_CACHE_CUTOFF)
        {
          if(psl.labels[u].vertices.size() <= SMART_DIST_CACHE_CUTOFF){
            new_labels[u] =  psl.Pull<false>(u, d, caches[tid], used[tid]);
            // cacheless_pull++;
          }
          else {
            new_labels[u] =  psl.Pull<true>(u, d, caches[tid], used[tid]);
            // cacheful_pull++;
          }
        } else {
          new_labels[u] =  psl.Pull<true>(u, d, caches[tid], used[tid]);
        }


        if (new_labels[u] != nullptr && !new_labels[u]->empty()) {
          updated = updated || true;
        }
      }
    }
    end = omp_get_wtime();
    PrintTime("Level " + to_string(d), end - start);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
    for (IDType i = 0; i < num_nodes; i++) {
      IDType u = nodes_to_process[i];
      if (!in_cut[u] && new_labels[u] != nullptr && !new_labels[u]->empty()) {
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

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
    for (IDType u = 0; u < csr.n;  u++) {
      auto &labels_u = psl.labels[u];
      labels_u.dist_ptrs.push_back(labels_u.vertices.size());

      if (new_labels[u] != nullptr) {
        delete new_labels[u];
        new_labels[u] = nullptr;
      }
     
      IDType labels_u_dist_size = labels_u.dist_ptrs.size(); 
      if(labels_u.dist_ptrs[labels_u_dist_size-2] != labels_u.dist_ptrs[labels_u_dist_size-1]){
        IDType start_neighbors = csr.row_ptr[u];
        IDType end_neighbors = csr.row_ptr[u + 1];
        for (IDType i = start_neighbors; i < end_neighbors; i++) {
          IDType v = csr.col[i];
          if (v < psl.max_ranks[u]) {
            should_run[v] = true;
          }
        }
      }

      psl.max_ranks[u] = -1;
    }

    // Stops the execution once all processes agree that they are done
    int updated_int = (int) updated;
    for (int i = 0; i < np; i++) {
      if(pid == i){
        BroadcastData(&updated_int, 1);
      } else {
        int *updated_other;
        RecvBroadcast(updated_other, i);
        updated_int |= *updated_other;
        delete updated_other;
      }
    }

#ifdef DEBUG
    psl.CountStats(omp_get_wtime() - alg_start);
#endif

    if (updated_int == 0) {
      break;
    }
  }

  delete[] nodes_to_process;

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

DPSL::DPSL(int pid, CSR *csr, int np, string vsep_file)
    : whole_csr(csr), pid(pid), np(np) {
  if (pid == 0) {
    InitP0(vsep_file);
  } else {
    Init();
  }
}

DPSL::~DPSL() {

  for (int i = 0; i < NUM_THREADS; i++) {
    delete[] caches[i];
  }
  delete[] caches;

  delete[] used;

  if(part_csr != nullptr)
    delete part_csr;

  // if(whole_csr != nullptr)
  //   delete whole_csr;

  delete psl_ptr;

  if (vc_ptr == nullptr)
    delete[] partition;
  else
    delete vc_ptr;

  if (global_bp != nullptr)
    delete global_bp;
}
