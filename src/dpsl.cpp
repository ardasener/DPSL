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
      size_t size = RecvData(dists, 0, p, MPI_INT32_T);
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

    delete[] counts;

    for (int p = 1; p < np; p++) {
      IDType *recv_counts;
      size_t size = RecvData(recv_counts, 0, p);
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




bool DPSL::MergeCut(vector<vector<IDType> *>& new_labels, PSL &psl) {
  
  // Keeps track of whether or not new labels were added to the vertices
  bool updated = false;

  size_t total_chunk_size = MERGE_CHUNK_SIZE * np;
  cout << "MERGE CHUNK SIZE: " << MERGE_CHUNK_SIZE << endl;
  cout << "TOTAL MERGE CHUNK SIZE: " << total_chunk_size << endl;

  // Outer Loop: Processes the data in rounds
  for(size_t round_start = 0; round_start < cut.size(); round_start += total_chunk_size){
    
    cout << "NEW ROUND: " << round_start << endl;

    vector<vector<IDType>> all_comp(MERGE_CHUNK_SIZE);
  
    IDType* comp_indices;
    IDType* comp_labels;

    for(int p=0; p<np; p++){
      size_t start = round_start + p * MERGE_CHUNK_SIZE;
      size_t end = start + MERGE_CHUNK_SIZE;
      
      if(p != pid){
        size_t num_vertices = CompressCutLabels(comp_indices, comp_labels, new_labels, start, end);
        SendData(comp_indices, num_vertices+1, MPI_LABEL_INDICES, p);
        SendData(comp_labels, comp_indices[num_vertices], MPI_LABELS, p);
        if(comp_indices != nullptr) delete[] comp_indices;
        if(comp_labels != nullptr) delete[] comp_labels;
      
      } else {
        
        size_t num_vertices = CompressCutLabels(comp_indices, comp_labels, new_labels, start, end);
#pragma omp parallel for num_threads(NUM_THREADS)
        for(size_t i=0; i<num_vertices; i++){
          size_t start = comp_indices[i];
          size_t end = comp_labels[i+1];
          all_comp[i].insert(all_comp[i].end(), comp_labels + start, comp_labels + end);
        }

        if(comp_indices != nullptr) delete[] comp_indices;
        if(comp_labels != nullptr) delete[] comp_labels;
      
        for(int p2=0; p2<np; p2++){
          if(p2 == pid) continue;
          size_t comp_indices_size = RecvData(comp_indices, MPI_LABEL_INDICES, p2);
          size_t comp_labels_size = RecvData(comp_labels, MPI_LABELS, p2);

          if(comp_indices_size == 0 || comp_labels_size == 0){
            continue;
          }

#pragma omp parallel for num_threads(NUM_THREADS)
          for(size_t i=0; i<comp_indices_size-1; i++){
            size_t start = comp_indices[i];
            size_t end = comp_labels[i+1];
            all_comp[i].insert(all_comp[i].end(), comp_labels + start, comp_labels + end);
          }

          if(comp_indices != nullptr) delete[] comp_indices;
          if(comp_labels != nullptr) delete[] comp_labels;

        }
      }
    }

    // Merge Operation
    vector<vector<bool>> seen(NUM_THREADS, vector<bool>(global_n, false));
#pragma omp parallel for num_threads(NUM_THREADS) 
    for(size_t i=0; i<MERGE_CHUNK_SIZE; i++){
      int tid = omp_get_thread_num();

      vector<IDType> merged;

      for(size_t j=0; j<all_comp[i].size(); j++){
        IDType u = all_comp[i][j];
        if(!seen[tid][u]){
          merged.push_back(u);
          seen[tid][u] = true;
        }
      }

      for(IDType u : merged){
        seen[tid][u] = false;
      }

      all_comp[i] = merged;
    }

    // Recompress the merged data
    IDType* merge_indices = new IDType[MERGE_CHUNK_SIZE + 1];

    fill(merge_indices, merge_indices + MERGE_CHUNK_SIZE + 1, 0);

    // Fill the size of each label_set in parallel
#pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t i=0; i < MERGE_CHUNK_SIZE; i++){
      merge_indices[i+1] = all_comp[i].size();
    }

    // Cumilate them (not in parallel)
    for(size_t i=1; i < MERGE_CHUNK_SIZE + 1; i++){
      merge_indices[i] += merge_indices[i-1];
    }


// Write the merged labels in parallel
  IDType* merge_labels = new IDType[merge_indices[MERGE_CHUNK_SIZE]];
#pragma omp parallel for num_threads(NUM_THREADS)
  for(size_t i=0; i < MERGE_CHUNK_SIZE; i++){ 
    size_t label_start_index = merge_indices[i];
    copy(all_comp[i].begin(), all_comp[i].end(), merge_labels + label_start_index);
  }

  // Broadcast and apply the labels
  for(int p=0; p<np; p++){
    IDType* recv_merge_indices;
    IDType* recv_merge_labels;

    // Recieve or broadcast the data depending on pid
    if(pid == p){
      BroadcastData(merge_indices, MERGE_CHUNK_SIZE+1);
      BroadcastData(merge_labels, merge_indices[MERGE_CHUNK_SIZE]);
      recv_merge_indices = merge_indices;
      recv_merge_labels = merge_labels;
    } else {
      RecvBroadcast(recv_merge_indices, p);
      RecvBroadcast(recv_merge_labels, p);
    }

    // Apply the received data to PSL
#pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t i=0; i < MERGE_CHUNK_SIZE; i++){
      size_t cut_index = round_start + p *MERGE_CHUNK_SIZE + i;

      if(cut_index >= cut.size()) continue;

      IDType u = cut[cut_index];

      IDType max_rank = -1;

      size_t start = recv_merge_indices[i];
      size_t end = recv_merge_indices[i+1];

      updated = updated || (end-start > 0);

      max_rank = max(max_rank, *max_element(recv_merge_labels + start, recv_merge_labels + end));
      psl.max_ranks[u] = max_rank;

      auto& labels_u = psl.labels[u].vertices;
      labels_u.insert(labels_u.end(), recv_merge_labels + start, recv_merge_labels + end);
    }

    // This will delete the received data
    // But note that on the broadcasting node it deletes the constructed data too
    if(recv_merge_indices != nullptr) delete[] recv_merge_indices;
    if(recv_merge_labels != nullptr) delete[] recv_merge_labels;

  } // Broadcast loop

  } // Outer loop

  return updated;

}

size_t DPSL::CompressCutLabels(IDType*& comp_indices, IDType*& comp_labels, vector<vector<IDType> *>& new_labels, 
  size_t start_index, size_t end_index){

  // Ensures we don't overflow the array
  start_index = min(start_index, cut.size());
  end_index = min(end_index, cut.size());
  
  size_t num_vertices = end_index - start_index;

  if(num_vertices == 0){
    comp_indices = nullptr;
    comp_labels == nullptr;
  }

  // Basically like the row_ptr array of a CSR  
  comp_indices = new IDType[num_vertices + 1];
  fill(comp_indices, comp_indices + num_vertices + 1, 0);

  // Fill the size of each label_set in parallel
#pragma omp parallel for num_threads(NUM_THREADS)
  for(size_t i=start_index; i < end_index; i++){
    IDType u = cut[i];
    size_t new_labels_size = (new_labels[u] != nullptr) ? new_labels[u]->size() : 0;
    comp_indices[i-start_index+1] = new_labels_size;
  }

  // Cumilate them (not in parallel)
  for(size_t i=1; i<num_vertices+1; i++){
    comp_indices[i] += comp_indices[i-1];
  }

  // Write the labels in parallel
  comp_labels = new IDType[comp_indices[num_vertices]];
 #pragma omp parallel for num_threads(NUM_THREADS)
  for(size_t i=start_index; i < end_index; i++){ 
    IDType u = cut[i];
    size_t label_start_index = comp_indices[i-start_index];
    copy(new_labels[u]->begin(), new_labels[u]->end(), comp_labels + label_start_index);
  }

  return num_vertices;
}




void DPSL::Barrier() { MPI_Barrier(MPI_COMM_WORLD); }

template <typename T>
void DPSL::SendData(T *data, size_t size, int tag, int to,
                           MPI_Datatype type) {
  int data_tag = (tag << 4);
  int size_tag = data_tag;

  MPI_Send(&size, 1, MPI_INT64_T, to, size_tag, MPI_COMM_WORLD);

  if (size != 0 && data != nullptr){

    int send_id = 1;
    while(size > MAX_COMM_SIZE){
      cout << "Large size data (" << size << ") in Send" << endl;
      int curr_tag = data_tag | send_id++;
      MPI_Send(data, MAX_COMM_SIZE, type, to, curr_tag, MPI_COMM_WORLD);
      data += MAX_COMM_SIZE;
      size -= MAX_COMM_SIZE;
    }

    if(size > 0){
      int curr_tag = data_tag | send_id++;
      MPI_Send(data, size, type, to, curr_tag, MPI_COMM_WORLD);
    }
  }
}

template <typename T>
void DPSL::BroadcastData(T *data, size_t size, MPI_Datatype type) {

  Barrier();
  MPI_Bcast(&size, 1, MPI_INT64_T, pid, MPI_COMM_WORLD);

  if(size != 0 && data != nullptr){
    while(size > MAX_COMM_SIZE){
      cout << "Large size data (" << size << ") in Broadcast" << endl;
      Barrier();
      MPI_Bcast(data, MAX_COMM_SIZE, type, pid, MPI_COMM_WORLD);
      data += MAX_COMM_SIZE;
      size -= MAX_COMM_SIZE;
    }

    if(size > 0){
      Barrier();
      MPI_Bcast(data, size, type, pid, MPI_COMM_WORLD); 
    }
  }
}

template <typename T>
size_t DPSL::RecvBroadcast(T *&data, int from, MPI_Datatype type){
  size_t size = 0;
  
  Barrier();
  MPI_Bcast(&size, 1, MPI_INT64_T, from, MPI_COMM_WORLD);

  size_t full_size = size;

  if(size != 0){
    data = new T[size];
    size_t sent = 0;
    while(size > MAX_COMM_SIZE){
      Barrier();
      MPI_Bcast(data + sent, MAX_COMM_SIZE, type, from, MPI_COMM_WORLD); 

      size -= MAX_COMM_SIZE;
      sent += MAX_COMM_SIZE;

    }

    if(size > 0){
      
      Barrier();
      MPI_Bcast(data + sent, size, type, from, MPI_COMM_WORLD); 

    }

  } else {
    data = nullptr;
  }
  return full_size;
}

template <typename T>
size_t DPSL::RecvData(T *&data, int tag, int from, MPI_Datatype type) {
  int data_tag = (tag << 4);
  int size_tag = data_tag;
  size_t size = 0;

  int error_code1, error_code2;

  MPI_Recv(&size, 1, MPI_INT64_T, from, size_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  size_t full_size = size;

  if (size != 0) {
    data = new T[size];

    int send_id = 1;
    size_t sent = 0;
    while(size > MAX_COMM_SIZE){
      int curr_tag = data_tag | send_id++;
      MPI_Recv(data + sent, MAX_COMM_SIZE, type, from, curr_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      size -= MAX_COMM_SIZE;
      sent += MAX_COMM_SIZE;
    }

    if(size > 0){
      int curr_tag = data_tag | send_id++;
      MPI_Recv(data + sent, size, type, from, curr_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

  } else {
    data = nullptr;
    Log("Recieved Size 0 Data");
  }

  return full_size;
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
    global_bp = new BP(csr, vc.ranks, vc.order, &cut);

    Log("Global BP Barrier Region");
    Barrier();
    vector<uint64_t> bp_sets(global_n*N_ROOTS*2);
    vector<uint8_t> bp_dists(global_n*N_ROOTS);

#pragma omp parallel default(shared) num_threads(NUM_THREADS)
    for (IDType i = 0; i < global_n; i++) {
      int offset1 = i*N_ROOTS;
      int offset2 = i*N_ROOTS*2;
      BPLabel &bp_label = global_bp->bp_labels[i];
      for (int j = 0; j < N_ROOTS; j++) {
        bp_dists[offset1 + j] = bp_label.bp_dists[j];
        bp_sets[offset2 + j*2] = bp_label.bp_sets[j][0]; 
        bp_sets[offset2 + j*2 + 1] = bp_label.bp_sets[j][1];
      }
    }

    cout << "Sending bp dists" << endl;
    BroadcastData(bp_dists.data(), bp_dists.size(), MPI_UINT8_T);
    cout << "Sending bp sets" << endl;
    BroadcastData(bp_sets.data(), bp_sets.size(), MPI_UINT64_T);

    int* bp_used = new int[global_n];

#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for(int i=0; i<global_n; i++){
      bp_used[i] = (int) global_bp->used[i];
    }
    cout << "Sending bp used" << endl;
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
  delete[] ranks_ptr;
  delete[] order_ptr;

  if constexpr (USE_GLOBAL_BP) {
    Log("Global BP Barrier Region");
    Barrier();
    vector<BPLabel> bp_labels(global_n);
  
    uint8_t *bp_dists;
    size_t bp_dists_size = RecvBroadcast(bp_dists, 0, MPI_UINT8_T);
    uint64_t *bp_sets;
    size_t bp_sets_size = RecvBroadcast(bp_sets, 0, MPI_UINT64_T);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for (IDType i = 0; i < global_n; i++) {
      int offset1 = i*N_ROOTS;
      int offset2 = i*N_ROOTS*2;
      for (IDType j = 0; j < N_ROOTS; j++) {
        bp_labels[i].bp_dists[j] = bp_dists[offset1 + j];
        bp_labels[i].bp_sets[j][0] = bp_sets[offset2 + j * 2];
        bp_labels[i].bp_sets[j][1] = bp_sets[offset2 + j * 2 + 1];
      }

    }

    delete[] bp_dists;
    delete[] bp_sets;

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

    global_bp = new BP(bp_labels, bp_used_vec);
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
        delete[] updated_other;
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

  delete[] partition;
  
  if (vc_ptr != nullptr)
    delete vc_ptr;

  if (global_bp != nullptr)
    delete global_bp;
}
