#include "dpsl.h"

void DPSL::Log(string msg) {
#ifdef DEBUG
  cout << "P" << pid << ": " << msg << endl;
#endif
}

void DPSL::PrintTime(string tag, double time) {
  cout << "P" << pid << ": " << tag << ", " << time << " seconds" << endl;
}

void DPSL::QueryTest(int query_count) {
  IDType *sources;
  if (pid == 0) {
    sources = new IDType[query_count];

    size_t cut_select = random_range(0, cut.size());
    sources[0] = cut[cut_select];
    for (int i = 1; i < query_count; i++) {
      sources[i] = random_range(0, whole_csr->n);
    }

    BroadcastData(sources, query_count);
  } else {
    RecvBroadcast(sources, 0);
  }

  for (int i = 0; i < query_count; i++) {
    Barrier();
    IDType u = sources[i];
    if (pid == 0)
      cout << "Query From: " << u << "(" << partition[u] << ")" << endl;
    Query(u, "output_dpsl_query_" + to_string(i) + ".txt");
  }

  delete[] sources;
}

void DPSL::Query(IDType u, string filename) {
  Log("Starting Query");
  Log("Global N: " + to_string(global_n));
  Barrier();

  double start_time, end_time;

  IDType u_comp = u;

  if constexpr (GLOBAL_COMPRESS || LOCAL_COMPRESS) {
    u_comp = part_csr->comp_ids[u];
  }

  IDType u_inv = part_csr->inv_ids[u_comp];

  int leaf_add_u = 0;
  if constexpr (ELIM_LEAF)
    if (psl_ptr->leaf_root[u_inv] != -1) {
      u_inv = psl_ptr->leaf_root[u_inv];
      leaf_add_u = 1;
    }
  // cout << "P" << pid << ": U: " << u << " " << u_inv << endl;

  start_time = omp_get_wtime();

  char *cache;

  if (partition[u_comp] == pid) {
    // cout << pid << " " << u << " " << u_comp << " " << u_inv << " " <<  (int)
    // part_csr->type[u] << endl;

    cache = new char[part_csr->n];
    fill(cache, cache + part_csr->n, -1);

    if (psl_ptr->local_min[u_inv]) {
      cache[u_inv] = 0 + leaf_add_u;

      IDType u_ngh_start = part_csr->row_ptr[u_inv];
      IDType u_ngh_end = part_csr->row_ptr[u_inv + 1];

      for (IDType i = u_ngh_start; i < u_ngh_end; i++) {
        auto &labels_un = psl_ptr->labels[part_csr->col[i]];
        for (int d = 0; d < last_dist; d++) {
          IDType dist_start = labels_un.dist_ptrs[d];
          IDType dist_end = labels_un.dist_ptrs[d + 1];

#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
          for (IDType j = dist_start; j < dist_end; j++) {
            IDType w = labels_un.vertices[j];

            if (cache[w] == -1)
              cache[w] = d + 1 + leaf_add_u;
            else
              cache[w] = min((int)cache[w], (int)d + 1 + leaf_add_u);
          }
        }
      }
    } else {
      auto &labels_u = psl_ptr->labels[u_inv];
#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
      for (IDType d = 0; d < last_dist; d++) {
        IDType start = labels_u.dist_ptrs[d];
        IDType end = labels_u.dist_ptrs[d + 1];

        for (IDType i = start; i < end; i++) {
          IDType v = labels_u.vertices[i];
          cache[v] = d + leaf_add_u;
        }
      }
    }

    Log("Broadcasting u's labels");
    BroadcastData(cache, part_csr->n, MPI_CHAR);
    if constexpr (ELIM_LEAF) BroadcastData(&leaf_add_u, 1, MPI_INT32_T);
  } else {
    Log("Recieving u's labels");
    RecvBroadcast(cache, partition[u_comp], MPI_CHAR);
    if constexpr (ELIM_LEAF) {
      int *temp;
      RecvBroadcast(temp, partition[u_comp], MPI_INT32_T);
      leaf_add_u = *temp;
      delete[] temp;
    }
  }

  Barrier();

  Log("Querying locally");
  vector<int> local_dist(part_csr->n, MAX_DIST);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
  for (IDType v = 0; v < part_csr->n; v++) {
    IDType v_comp = v;

    if constexpr (GLOBAL_COMPRESS || LOCAL_COMPRESS) {
      v_comp = part_csr->comp_ids[v];
    }

    if (partition[v_comp] != pid) continue;

    if (u == v) {
      local_dist[v] = 0;
      continue;
    }

    IDType v_inv = part_csr->inv_ids[v_comp];

    // if(v == 119) cout << partition[v] <<  " " << v << " "  << v_comp << " "
    // <<  v_inv << " " << (int) part_csr->type[v] << endl; if(v == 36) cout <<
    // partition[v] <<  " " << v << " "  << v_comp << " " <<  v_inv << " " <<
    // (int) part_csr->type[v] << endl;

    if constexpr (GLOBAL_COMPRESS || LOCAL_COMPRESS)
      if (v_comp == u_comp) {
        local_dist[v] = max(part_csr->type[u], part_csr->type[v]);
        continue;
      }

    int leaf_add_v = 0;
    if constexpr (ELIM_LEAF)
      if (psl_ptr->leaf_root[v_inv] != -1) {
        // if(v == 2524) cout << v << " " << v_inv << " " <<
        // psl_ptr->leaf_root[v_inv] << " | " << u << " " << u_inv << " " <<
        // endl;

        if (psl_ptr->leaf_root[v_inv] == u_inv) {
          local_dist[v] = 1 + leaf_add_u;
          continue;
        }

        v_inv = psl_ptr->leaf_root[v_inv];
        leaf_add_v = 1;
      }

    int min_dist = MAX_DIST;

    if (cache[v_inv] != -1) {
      // if(v == 2524) cout << v << " " << v_inv << " " << (int) cache[v_inv] <<
      // endl;
      min_dist = min((int)cache[v_inv] + leaf_add_v, min_dist);
    }

    if constexpr (USE_GLOBAL_BP) {
      int global_bp_dist = global_bp->QueryByBp(part_csr->inv_ids[u_comp],
                                                part_csr->inv_ids[v_comp]);
      min_dist = min(min_dist, global_bp_dist);
    }

    if constexpr (USE_LOCAL_BP) {
      int local_bp_dist =
          psl_ptr->local_bp->QueryByBp(u_inv, v_inv) + leaf_add_v + leaf_add_u;
      min_dist = min(min_dist, local_bp_dist);
    }

    if (psl_ptr->local_min[v_inv]) {
      IDType v_ngh_start = part_csr->row_ptr[v_inv];
      IDType v_ngh_end = part_csr->row_ptr[v_inv + 1];
      for (int d = 0; d + 1 < min_dist && d < last_dist; d++) {
        for (IDType i = v_ngh_start; i < v_ngh_end; i++) {
          auto &labels_vn = psl_ptr->labels[part_csr->col[i]];
          IDType dist_start = labels_vn.dist_ptrs[d];
          IDType dist_end = labels_vn.dist_ptrs[d + 1];

          for (IDType j = dist_start; j < dist_end; j++) {
            IDType w = labels_vn.vertices[j];

            if (cache[w] == -1) {
              continue;
            }

            int dist = d + 1 + (int)cache[w] + leaf_add_v;
            min_dist = min(min_dist, dist);
          }
        }
      }
    } else {
      auto &labels_v = psl_ptr->labels[v_inv];

      // if(v == 14){
      //       cout << "P" << pid << ":" << v_inv << endl;
      //     }

      for (int d = 0; d < last_dist && d < min_dist; d++) {
        IDType start = labels_v.dist_ptrs[d];
        IDType end = labels_v.dist_ptrs[d + 1];

        for (IDType i = start; i < end; i++) {
          IDType w = labels_v.vertices[i];

          if (cache[w] == -1) {
            continue;
          }

          int dist = d + (int)cache[w] + leaf_add_v;
          min_dist = min(dist, min_dist);

          // if(v == 14){
          //   cout << "P" << pid << ":" << min_dist << endl;
          // }
        }
      }
    }

    local_dist[v] = min_dist;
  }

  delete[] cache;

  Barrier();

  Log("Synchronizing query results");
  if (pid == 0) {
    int *all_dists = new int[whole_csr->n];
    int *source = new int[whole_csr->n];
    fill(all_dists, all_dists + whole_csr->n, MAX_DIST);
    fill(source, source + whole_csr->n, 0);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) \
    schedule(SCHEDULE)
    for (IDType i = 0; i < local_dist.size(); i++) {
      all_dists[i] = local_dist[i];
    }

    for (int p = 1; p < np; p++) {
      int *dists;
      size_t size = RecvData(dists, 0, p, MPI_INT32_T);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) \
    schedule(SCHEDULE)
      for (int i = 0; i < whole_csr->n; i++) {
        if (dists[i] < all_dists[i]) {
          source[i] = p;
          all_dists[i] = dists[i];
        }
      }
      delete[] dists;
    }

    end_time = omp_get_wtime();

    // cout << "Avg. Query Time: " << (end_time-start_time) / whole_csr->n << "
    // seconds" << endl;

    start_time = omp_get_wtime();
    vector<int> *bfs_results = BFSQuery(*unordered_csr, u);
    end_time = omp_get_wtime();

    // cout << "Avg. BFS Time: " << (end_time-start_time) / whole_csr->n << "
    // seconds" << endl;;

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

      if (bfs_res != psl_res) {
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
         -1);  // -1 indicates free floating vertex

    for (IDType i = 0; i < part_csr->n; i++) {
      all_counts[i] = counts[i];
      source[i] = 0;  // 0 indicates cut vertex as well as partition 0
    }

    delete[] counts;

    for (int p = 1; p < np; p++) {
      IDType *recv_counts;
      size_t size = RecvData(recv_counts, 0, p);
      for (IDType i = 0; i < size; i++) {
        if (recv_counts[i] != -1) {  // Count recieved
          if (vc_ptr->cut.find(i) == vc_ptr->cut.end() &&
              all_counts[i] < recv_counts[i]) {  // vertex not in cut and
                                                 // counted on the recieved data

            all_counts[i] = recv_counts[i];  // Update count
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
        for (int j = 0; j < np; j++) total_per_source[j] += all_counts[i];
      }
    }

#ifdef DEBUG
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
    ofs << "Avg. Label Count: " << total / (double)whole_csr->n << endl;
    for (int p = 0; p < np; p++) {
      ofs << "Total for P" << p << ": " << total_per_source[p] << endl;
      ofs << "Avg for P" << p << ": "
          << total_per_source[p] / (double)whole_csr->n << endl;
    }

    ofs << endl;
    ofs.close();

    cout << "Total Label Count: " << total << endl;
    cout << "Avg. Label Count: " << total / (double)whole_csr->n << endl;

#endif
    


    delete[] all_counts;
    delete[] source;
    delete[] total_per_source;
  }
}

size_t DPSL::MergeCut(vector<vector<IDType> *> &new_labels, PSL &psl) {
  Barrier();

  size_t labels_added = 0;

  // Outer Loop: Processes the data in rounds
  for (size_t round_start = 0; round_start < cut.size();
       round_start += MERGE_CHUNK_SIZE) {
    // cout << "NEW ROUND: " << round_start << endl;
    size_t round_size = min((size_t)MERGE_CHUNK_SIZE, cut.size() - round_start);
    size_t per_node_chunk_size = (round_size / np) + 1;

    vector<vector<IDType>> all_comp(per_node_chunk_size);

    IDType *send_comp_indices;
    IDType *send_comp_labels;
    IDType *recv_comp_indices;
    IDType *recv_comp_labels;
    size_t recv_comp_indices_size;
    size_t recv_comp_labels_size;

    size_t self_start = round_start + pid * per_node_chunk_size;
    size_t self_end = self_start + per_node_chunk_size;

#pragma omp parallel for num_threads(NUM_THREADS) schedule(SCHEDULE)
    for (size_t i = self_start; i < self_end; i++) {
      if (i < cut_merge_order.size()) {
        IDType u = cut_merge_order[i];
        if (new_labels[u] != nullptr)
          all_comp[i - self_start].insert(all_comp[i - self_start].end(),
                                          new_labels[u]->begin(),
                                          new_labels[u]->end());
      }
    }

    // Used for round-robin communication
    // For 8 nodes initially we would have:
    // 0 1 2 3
    // 7 6 5 4
    vector<int> processes(np);
    iota(processes.begin(), processes.end(), 0);

    for (int k = 0; k < np - 1; k++) {
      // Everyone pairs up with the node accross them in the representation
      // shown above
      int self_index =
          find(processes.begin(), processes.end(), pid) - processes.begin();
      int pair = processes[(np - self_index - 1) % np];

      // Rotate the vector to the right except the first element
      rotate(processes.rbegin(), processes.rbegin() + 1, processes.rend() - 1);

      // cout << "K: " << k << " PID: " << pid << " Pair: " << pair << endl;

      size_t start = round_start + pair * per_node_chunk_size;
      size_t end = start + per_node_chunk_size;

      size_t num_vertices = CompressCutLabels(
          send_comp_indices, send_comp_labels, new_labels, start, end);

      // The process with the smaller pid sends first
      if (pid < pair) {
        SendData(send_comp_indices, num_vertices + 1, MPI_LABEL_INDICES, pair);
        SendData(send_comp_labels, send_comp_indices[num_vertices], MPI_LABELS,
                 pair);
        recv_comp_indices_size =
            RecvData(recv_comp_indices, MPI_LABEL_INDICES, pair);
        recv_comp_labels_size = RecvData(recv_comp_labels, MPI_LABELS, pair);

      } else {
        recv_comp_indices_size =
            RecvData(recv_comp_indices, MPI_LABEL_INDICES, pair);
        recv_comp_labels_size = RecvData(recv_comp_labels, MPI_LABELS, pair);
        SendData(send_comp_indices, num_vertices + 1, MPI_LABEL_INDICES, pair);
        SendData(send_comp_labels, send_comp_indices[num_vertices], MPI_LABELS,
                 pair);
      }

      delete[] send_comp_indices;
      delete[] send_comp_labels;

#pragma omp parallel for num_threads(NUM_THREADS) schedule(SCHEDULE)
      for (size_t i = 0; i < recv_comp_indices_size - 1; i++) {
        size_t start = recv_comp_indices[i];
        size_t end = recv_comp_indices[i + 1];
        all_comp[i].insert(all_comp[i].end(), recv_comp_labels + start,
                           recv_comp_labels + end);
      }

      delete[] recv_comp_indices;
      delete[] recv_comp_labels;
    }

    // Merge Operation
    // cout << "Merging P" << pid << endl;
#pragma omp parallel for num_threads(NUM_THREADS) schedule(SCHEDULE)
    for (size_t i = 0; i < per_node_chunk_size; i++) {
      int tid = omp_get_thread_num();

      vector<IDType> merged;

      for (size_t j = 0; j < all_comp[i].size(); j++) {
        IDType u = all_comp[i][j];
        if (!merge_seen[tid][u]) {
          merged.push_back(u);
          merge_seen[tid][u] = true;
        }
      }

      for (IDType u : merged) {
        merge_seen[tid][u] = false;
      }

      all_comp[i].clear();
      all_comp[i] = move(merged);
    }
    // cout << "DONE Merging P" << pid << endl;

    // Recompress the merged data
    // cout << "Recompress P" << pid << endl;
    IDType *merge_indices = new IDType[per_node_chunk_size + 1];
    merge_indices[0] = 0;

    // Fill the size of each label_set in parallel
#pragma omp parallel for num_threads(NUM_THREADS) schedule(SCHEDULE)
    for (size_t i = 0; i < per_node_chunk_size; i++) {
      merge_indices[i + 1] = all_comp[i].size();
    }

    // Cumilate them (not in parallel)
    for (size_t i = 1; i < per_node_chunk_size + 1; i++) {
      merge_indices[i] += merge_indices[i - 1];
    }

    // Write the merged labels in parallel
    IDType *merge_labels = new IDType[merge_indices[per_node_chunk_size]];
#pragma omp parallel for num_threads(NUM_THREADS) schedule(SCHEDULE)
    for (size_t i = 0; i < per_node_chunk_size; i++) {
      size_t label_start_index = merge_indices[i];
      copy(all_comp[i].begin(), all_comp[i].end(),
           merge_labels + label_start_index);
    }

    // cout << "DONE Recompress P" << pid << endl;

    Barrier();


    // Broadcast and apply the labels
    for (int p = 0; p < np; p++) {
      IDType *recv_merge_indices;
      IDType *recv_merge_labels;

      // Recieve or broadcast the data depending on pid
      if (pid == p) {
        // cout << "Broadcasting P" << pid << endl;
        BroadcastData(merge_indices, per_node_chunk_size + 1);
        BroadcastData(merge_labels, merge_indices[per_node_chunk_size]);
        recv_merge_indices = merge_indices;
        recv_merge_labels = merge_labels;
        labels_added += merge_indices[per_node_chunk_size];
        // cout << "DONE Broadcasting P" << pid << endl;
      } else {
        // cout << "Recv. Broadcast P" << pid << endl;
        size_t s1 = RecvBroadcast(recv_merge_indices, p);
        size_t s2 = RecvBroadcast(recv_merge_labels, p);
        labels_added += s2;
        // cout << "DONE Recv. Broadcast P" << pid << " with size=" << s1 << ",
        // " << s2 << endl;
      }



      // Apply the received data to PSL
      // cout << "Apply data P" << pid << endl;
#pragma omp parallel for num_threads(NUM_THREADS) schedule(SCHEDULE)
      for (size_t i = 0; i < per_node_chunk_size; i++) {
        size_t cut_index = round_start + p * per_node_chunk_size + i;

        if (cut_index >= cut_merge_order.size()) continue;

        IDType u = cut_merge_order[cut_index];

        IDType max_rank = -1;

        size_t start = recv_merge_indices[i];
        size_t end = recv_merge_indices[i + 1];

        if (end - start > 0) {
          max_rank =
              *max_element(recv_merge_labels + start, recv_merge_labels + end);
          auto &labels_u = psl.labels[u].vertices;
          sort(recv_merge_labels + start, recv_merge_labels + end,
               greater<IDType>());
          labels_u.insert(labels_u.end(), recv_merge_labels + start,
                          recv_merge_labels + end);
        }

        psl.max_ranks[u] = max_rank;
      }
      // cout << "DONE Apply data P" << pid << endl;

      // This will delete the received data
      // But note that on the broadcasting node it deletes the constructed data
      // too
      if (recv_merge_indices != nullptr) delete[] recv_merge_indices;
      if (recv_merge_labels != nullptr) delete[] recv_merge_labels;

    }  // Broadcast loop

  }  // Outer loop

  return labels_added;
}

size_t DPSL::CompressCutLabels(IDType *&comp_indices, IDType *&comp_labels,
                               vector<vector<IDType> *> &new_labels,
                               size_t start_index, size_t end_index) {
  // Ensures we don't overflow the array
  start_index = min(start_index, cut_merge_order.size());
  end_index = min(end_index, cut_merge_order.size());

  size_t num_vertices = end_index - start_index;

  // cout << "Compress: " << start_index << ", " << end_index << ", " <<
  // num_vertices << endl;

  if (num_vertices == 0) {
    comp_indices = nullptr;
    comp_labels == nullptr;
  }

  // Basically like the row_ptr array of a CSR
  comp_indices = new IDType[num_vertices + 1];
  comp_indices[0] = 0;

  // Fill the size of each label_set in parallel
  // cout << "Filling sizes" << endl;
#pragma omp parallel for num_threads(NUM_THREADS) schedule(SCHEDULE)
  for (size_t i = start_index; i < end_index; i++) {
    IDType u = cut_merge_order[i];
    size_t new_labels_size =
        (new_labels[u] != nullptr) ? new_labels[u]->size() : 0;
    comp_indices[i - start_index + 1] = new_labels_size;
  }
  // cout << "DONE Filling sizes" << endl;

  // Cumilate them (not in parallel)
  // cout << "Sum sizes" << endl;
  for (size_t i = 1; i < num_vertices + 1; i++) {
    comp_indices[i] += comp_indices[i - 1];
  }
  // cout << "DONE Sum sizes" << endl;

  // Write the labels in parallel
  // cout << "Write labels" << endl;
  comp_labels = new IDType[comp_indices[num_vertices]];
#pragma omp parallel for num_threads(NUM_THREADS) schedule(SCHEDULE)
  for (size_t i = start_index; i < end_index; i++) {
    IDType u = cut_merge_order[i];
    size_t label_start_index = comp_indices[i - start_index];
    if (new_labels[u] != nullptr)
      copy(new_labels[u]->begin(), new_labels[u]->end(),
           comp_labels + label_start_index);
  }
  // cout << "DONE Write labels" << endl;

  return num_vertices;
}

void DPSL::Barrier() { MPI_Barrier(MPI_COMM_WORLD); }

template <typename T>
void DPSL::SendData(T *data, size_t size, int tag, int to, MPI_Datatype type) {
  int data_tag = (tag << 4);
  int size_tag = data_tag;

  MPI_Send(&size, 1, MPI_INT64_T, to, size_tag, MPI_COMM_WORLD);

  if (size != 0 && data != nullptr) {
    int send_id = 1;
    while (size > MAX_COMM_SIZE) {
      cout << "Large size data (" << size << ") in Send" << endl;
      int curr_tag = data_tag | send_id++;
      MPI_Send(data, MAX_COMM_SIZE, type, to, curr_tag, MPI_COMM_WORLD);
      data += MAX_COMM_SIZE;
      size -= MAX_COMM_SIZE;
    }

    if (size > 0) {
      int curr_tag = data_tag | send_id++;
      MPI_Send(data, size, type, to, curr_tag, MPI_COMM_WORLD);
    }
  }
}

template <typename T>
void DPSL::BroadcastData(T *data, size_t size, MPI_Datatype type) {
  Barrier();
  MPI_Bcast(&size, 1, MPI_INT64_T, pid, MPI_COMM_WORLD);

  if (size != 0 && data != nullptr) {
    while (size > MAX_COMM_SIZE) {
      cout << "Large size data (" << size << ") in Broadcast" << endl;
      Barrier();
      MPI_Bcast(data, MAX_COMM_SIZE, type, pid, MPI_COMM_WORLD);
      data += MAX_COMM_SIZE;
      size -= MAX_COMM_SIZE;
    }

    if (size > 0) {
      Barrier();
      MPI_Bcast(data, size, type, pid, MPI_COMM_WORLD);
    }
  }
}

template <typename T>
size_t DPSL::RecvBroadcast(T *&data, int from, MPI_Datatype type) {
  size_t size = 0;

  Barrier();
  MPI_Bcast(&size, 1, MPI_INT64_T, from, MPI_COMM_WORLD);

  size_t full_size = size;

  if (size != 0) {
    data = new T[size];
    size_t sent = 0;
    while (size > MAX_COMM_SIZE) {
      Barrier();
      MPI_Bcast(data + sent, MAX_COMM_SIZE, type, from, MPI_COMM_WORLD);

      size -= MAX_COMM_SIZE;
      sent += MAX_COMM_SIZE;
    }

    if (size > 0) {
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

  MPI_Recv(&size, 1, MPI_INT64_T, from, size_tag, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  size_t full_size = size;

  if (size != 0) {
    data = new T[size];

    int send_id = 1;
    size_t sent = 0;
    while (size > MAX_COMM_SIZE) {
      int curr_tag = data_tag | send_id++;
      MPI_Recv(data + sent, MAX_COMM_SIZE, type, from, curr_tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      size -= MAX_COMM_SIZE;
      sent += MAX_COMM_SIZE;
    }

    if (size > 0) {
      int curr_tag = data_tag | send_id++;
      MPI_Recv(data + sent, size, type, from, curr_tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

  } else {
    data = nullptr;
    Log("Recieved Size 0 Data");
  }

  return full_size;
}

void DPSL::InitP0(string partition_str, string partition_params) {
  double init_start = omp_get_wtime();

  string order_method = ORDER_METHOD;
  CSR &csr = *whole_csr;
  global_n = csr.n;

  unordered_csr = new CSR(csr);

  // cout << "Pre-Compression" << endl;
  // csr.PrintMetadata();

  if (GLOBAL_COMPRESS) {
    csr.Compress();
  }


  // cout << "Post-Compression" << endl;
  // csr.PrintMetadata();

  if (partition_str == "")
    throw "Partition file or partitioner is required";
  else if (partition_str.find(".part") != string::npos)
    vc_ptr = VertexCut::Read(csr, partition_str, order_method, np);
  else
    vc_ptr = VertexCut::Partition(csr, partition_str, partition_params,
                                  order_method, np);

  VertexCut &vc = *vc_ptr;

  Log("Creating All Cut");
  cut.insert(cut.end(), vc.cut.begin(), vc.cut.end());

  in_cut.resize(global_n, false);
  for (IDType u : cut) {
    in_cut[u] = true;
  }

  ranks = vc.ranks;
  order = vc.order;

  Log("Ordering Cut By Rank");
  sort(cut.begin(), cut.end(),
       [this](IDType u, IDType v) { return ranks[u] < ranks[v]; });

  auto &csrs = vc.csrs;

  csr.Reorder(order, &cut, &in_cut);

  for (int p = 0; p < np; p++) csrs[p]->Reorder(order, &cut, &in_cut);



  part_csr = csrs[0];

#ifdef DEBUG
  ofstream cut_ofs("output_dpsl_cut.txt");
  for (int i = 0; i < cut.size(); i++) {
    cut_ofs << cut[i] << endl;
  }
  cut_ofs.close();
#endif

  // #pragma omp parallel default(shared) num_threads(NUM_THREADS)
  for (int i = 0; i < cut.size(); i++) {
    cut[i] = csr.inv_ids[cut[i]];
  }

  in_cut.clear();
  in_cut.resize(global_n, false);
  for (IDType u : cut) {
    in_cut[u] = true;
  }


  bool* local_min_arr = new bool[csr.n];
  fill(local_min_arr, local_min_arr + csr.n, false);
  IDType* leaf_root_arr = new IDType[csr.n];
  fill(leaf_root_arr, leaf_root_arr + csr.n, -1);

  size_t leaf_count = 0;
  size_t local_min_count = 0;

  if constexpr (ELIM_LEAF) {
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE) reduction(+ : leaf_count)
    for (IDType u = 0; u < csr.n; u++) {
      if (csr.row_ptr[u] + 1 == csr.row_ptr[u + 1]) {
        IDType root = csr.col[csr.row_ptr[u]];

        if (csr.row_ptr[root] + 1 < csr.row_ptr[root + 1] && csr.type[u] == 0) {
          leaf_root_arr[u] = root;
          leaf_count++;
        }
      }
    }
    cout << "Found and eliminated " << leaf_count << " leaves" << endl;
  }


  if constexpr (ELIM_MIN) {
  #pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE) reduction(+ : local_min_count)
      for (IDType u = 0; u < csr.n; u++) {
        // if (in_cut[u]) continue;
        if (leaf_root_arr[u] != -1) continue;

        IDType start = csr.row_ptr[u];
        IDType end = csr.row_ptr[u + 1];

        if (end == start) {
          local_min_arr[u] = true;
          local_min_count++;
          continue;
        }

        for (IDType i = end - 1; i >= start; i--) {
          IDType v = csr.col[i];

          if (u >= v && leaf_root_arr[v] == -1) {
            break;
          }

          if (leaf_root_arr[v] == -1) {
            local_min_arr[u] = true;
            local_min_count++;
            break;
          }
        }
      }
      cout << "Found and eliminated " << local_min_count << " local min vertices"
          << endl;
    }

  local_min.resize(csr.n);
  copy(local_min_arr, local_min_arr + csr.n, local_min.begin());
  leaf_root.resize(csr.n);
  copy(leaf_root_arr, leaf_root_arr + csr.n, leaf_root.begin());


  cut_merge_order = cut;
  random_shuffle(cut_merge_order.begin(), cut_merge_order.end());

  partition = new IDType[csr.n];
  for (int i = 0; i < csr.n; i++) {
    // IDType new_id = csr.inv_ids[i];
    partition[i] = vc.partition[i];
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
    SendData(csrs[i]->ids, csrs[i]->n, MPI_CSR_IDS, i);
    SendData(csrs[i]->inv_ids, csrs[i]->n, MPI_CSR_INV_IDS, i);
    SendData(csrs[i]->type, csrs[i]->n, MPI_CSR_TYPE, i, MPI_CHAR);
    SendData(csrs[i]->comp_ids, csrs[i]->n, MPI_CSR_COMP_IDS, i);
    SendData(partition, csr.n, MPI_PARTITION, i);
    SendData(cut.data(), cut.size(), MPI_CUT, i);
    SendData(cut_merge_order.data(), cut_merge_order.size(),
             MPI_CUT_MERGE_ORDER, i);
    SendData(ranks.data(), ranks.size(), MPI_VERTEX_RANKS, i);
    SendData(order.data(), order.size(), MPI_VERTEX_ORDER, i);
    SendData(leaf_root_arr, csrs[i]->n, MPI_LEAF_ROOT, i);
    SendData(local_min_arr, csrs[i]->n, MPI_LOCAL_MIN, i, MPI_CXX_BOOL);

    // cout << "Post-Partition P" << i << endl;
    // csrs[i]->PrintMetadata();

    delete csrs[i];
    csrs[i] = nullptr;
  }

  delete[] leaf_root_arr;
  delete[] local_min_arr;

  Barrier();
  Log("Initial Barrier Region End");

  if constexpr (USE_GLOBAL_BP) {
    Log("Creating Global BP");
    global_bp = new BP(csr, vc.ranks, vc.order, &cut);

    Log("Global BP Barrier Region");
    Barrier();
    vector<uint64_t> bp_sets(global_n * N_ROOTS * 2);
    vector<uint8_t> bp_dists(global_n * N_ROOTS);

#pragma omp parallel default(shared) num_threads(NUM_THREADS)
    for (IDType i = 0; i < global_n; i++) {
      int offset1 = i * N_ROOTS;
      int offset2 = i * N_ROOTS * 2;
      BPLabel &bp_label = global_bp->bp_labels[i];
      for (int j = 0; j < N_ROOTS; j++) {
        bp_dists[offset1 + j] = bp_label.bp_dists[j];
        bp_sets[offset2 + j * 2] = bp_label.bp_sets[j][0];
        bp_sets[offset2 + j * 2 + 1] = bp_label.bp_sets[j][1];
      }
    }

    cout << "Sending bp dists" << endl;
    BroadcastData(bp_dists.data(), bp_dists.size(), MPI_UINT8_T);
    cout << "Sending bp sets" << endl;
    BroadcastData(bp_sets.data(), bp_sets.size(), MPI_UINT64_T);

    int *bp_used = new int[global_n];

#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for (int i = 0; i < global_n; i++) {
      bp_used[i] = (int)global_bp->used[i];
    }
    cout << "Sending bp used" << endl;
    BroadcastData(bp_used, global_n);
    delete[] bp_used;

    Barrier();
    Log("Global BP Barrier Region End");
  }

  Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));
  Log("Cut Size: " + to_string(cut.size()));

  merge_seen.resize(NUM_THREADS, vector<bool>(csr.n, false));

  double init_end = omp_get_wtime();
  double comm_end = omp_get_wtime();

  cout << "Init Total: " << init_end - init_start << " seconds" << endl;
  cout << "Init Communication: " << comm_end - comm_start << " seconds" << endl;
}

void DPSL::Init() {
  IDType *row_ptr;
  IDType *col;
  IDType *ids;
  IDType *inv_ids;
  char *type;
  IDType *comp_ids;
  IDType *cut_ptr;
  IDType *cut_merge_order_ptr;
  IDType *ranks_ptr;
  IDType *order_ptr;
  IDType *leaf_root_arr; 
  bool *local_min_arr; 

  Log("Initial Barrier Region");
  Barrier();

  IDType *global_n_ptr;
  RecvData(global_n_ptr, MPI_GLOBAL_N, 0);
  global_n = *global_n_ptr;
  delete[] global_n_ptr;

  IDType size_row_ptr = RecvData(row_ptr, MPI_CSR_ROW_PTR, 0);
  IDType size_col = RecvData(col, MPI_CSR_COL, 0);
  IDType size_ids = RecvData(ids, MPI_CSR_IDS, 0);
  IDType size_inv_ids = RecvData(inv_ids, MPI_CSR_INV_IDS, 0);
  IDType size_type = RecvData(type, MPI_CSR_TYPE, 0, MPI_CHAR);
  IDType size_comp_ids = RecvData(comp_ids, MPI_CSR_COMP_IDS, 0);
  IDType size_partition = RecvData(partition, MPI_PARTITION, 0);
  IDType size_cut = RecvData(cut_ptr, MPI_CUT, 0);
  IDType size_cut_merge_order =
      RecvData(cut_merge_order_ptr, MPI_CUT_MERGE_ORDER, 0);
  IDType size_ranks = RecvData(ranks_ptr, MPI_VERTEX_RANKS, 0);
  IDType size_order = RecvData(order_ptr, MPI_VERTEX_ORDER, 0);
  IDType size_leaf_root = RecvData(leaf_root_arr, MPI_LEAF_ROOT, 0);
  IDType size_local_min = RecvData(local_min_arr, MPI_LOCAL_MIN, 0, MPI_CXX_BOOL);
  Barrier();
  Log("Initial Barrier Region End");

  ranks.insert(ranks.end(), ranks_ptr, ranks_ptr + size_ranks);
  order.insert(order.end(), order_ptr, order_ptr + size_order);
  delete[] ranks_ptr;
  delete[] order_ptr;

  local_min.resize(size_local_min);
  copy(local_min_arr, local_min_arr + size_local_min, local_min.begin());
  leaf_root.resize(size_leaf_root);
  copy(leaf_root_arr, leaf_root_arr + size_leaf_root, leaf_root.begin());

  cut_merge_order.insert(cut_merge_order.end(), cut_merge_order_ptr,
                         cut_merge_order_ptr + size_cut);
  delete[] cut_merge_order_ptr;

  cut.insert(cut.end(), cut_ptr, cut_ptr + size_cut);
  in_cut.resize(global_n, false);
  for (IDType u : cut) {
    in_cut[u] = true;
  }
  delete[] cut_ptr;

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
      int offset1 = i * N_ROOTS;
      int offset2 = i * N_ROOTS * 2;
      for (IDType j = 0; j < N_ROOTS; j++) {
        bp_labels[i].bp_dists[j] = bp_dists[offset1 + j];
        bp_labels[i].bp_sets[j][0] = bp_sets[offset2 + j * 2];
        bp_labels[i].bp_sets[j][1] = bp_sets[offset2 + j * 2 + 1];
      }
    }

    delete[] bp_dists;
    delete[] bp_sets;

    int *bp_used;
    size_t bp_used_size = RecvBroadcast(bp_used, 0);

    vector<bool> bp_used_vec(global_n);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for (int i = 0; i < global_n; i++) {
      bp_used_vec[i] = (bool)bp_used[i];
    }

    delete[] bp_used;
    Barrier();
    Log("Global BP Barrier Region End");

    global_bp = new BP(bp_labels, bp_used_vec);
  }

  part_csr = new CSR(row_ptr, col, ids, inv_ids, type, comp_ids,
                     size_row_ptr - 1, size_col);
  Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));

  merge_seen.resize(NUM_THREADS, vector<bool>(part_csr->n, false));
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
  for (int i = 0; i < NUM_THREADS; i++) {
    used[i].resize(csr.n, false);
  }

  bool should_run[csr.n];
  fill(should_run, should_run + csr.n, true);

  string order_method = ORDER_METHOD;

  vector<IDType> *ranks_ptr = &ranks;
  vector<IDType> *order_ptr = &order;

  psl_ptr =
      new PSL(*part_csr, order_method, &cut, global_bp, ranks_ptr, order_ptr);
  PSL &psl = *psl_ptr;

  psl.leaf_root = leaf_root;
  psl.local_min = local_min;

  size_t l0_labels = 0;

  start = omp_get_wtime();
  alg_start = omp_get_wtime();
  vector<vector<IDType> *> init_labels(csr.n, nullptr);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE) reduction(+ : l0_labels)
  for (IDType u = 0; u < csr.n; u++) {
    bool should_init = true;
    if constexpr (USE_GLOBAL_BP) {
      if (global_bp->used[u]) should_init = false;
    }

    if constexpr (ELIM_MIN)
      if (psl.local_min[u]) should_init = false;

    if constexpr (ELIM_LEAF)
      if (psl.leaf_root[u] != -1) should_init = false;

    if (should_init) {
      psl.labels[u].vertices.push_back(u);
      init_labels[u] = psl.Init(u);
      l0_labels++;
    }
  }

  size_t l1_labels = 0;

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE) reduction(+ : l1_labels)
  for (IDType u = 0; u < csr.n; u++) {
    if (!in_cut[u] && init_labels[u] != nullptr && !init_labels[u]->empty()) {
      auto &labels = psl.labels[u].vertices;
      labels.insert(labels.end(), init_labels[u]->begin(),
                    init_labels[u]->end());
      l1_labels += init_labels[u]->size();
      delete init_labels[u];
      init_labels[u] = nullptr;
    }
  }
  end = omp_get_wtime();
  PrintTime("Level 0&1", end - start);

  Barrier();
  Log("Merging Initial Labels");
  start = omp_get_wtime();
  MergeCut(init_labels, psl);
  end = omp_get_wtime();
  PrintTime("Merge 0&1", end - start);
  total_merge_time += end - start;
  Log("Merging Initial Labels End");

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
  for (IDType u = 0; u < csr.n; u++) {
    auto &labels = psl.labels[u];

    bool should_init = true;
    if constexpr (USE_GLOBAL_BP)
      if (global_bp->used[u]) should_init = false;

    if constexpr (ELIM_MIN)
      if (psl.local_min[u]) should_init = false;

    if constexpr (ELIM_LEAF)
      if (psl.leaf_root[u] != -1) should_init = false;

    if (should_init) {
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(1);
      labels.dist_ptrs.push_back(labels.vertices.size());

      IDType ngh_start = csr.row_ptr[u];
      IDType ngh_end = csr.row_ptr[u + 1];

      for (IDType i = ngh_start; i < ngh_end; i++) {
        IDType v = csr.col[i];

        if (v < psl.max_ranks[u]) should_run[v] = true;
      }

      psl.prev_max_ranks[u] = psl.max_ranks[u];
      psl.max_ranks[u] = -1;

      delete init_labels[u];
    } else {
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(0);
    }
  }


  cout << "P" << pid << ": " << "Level 0&1 Count: " << l0_labels << "," << l1_labels << endl; 

  IDType *nodes_to_process = new IDType[csr.n];
  IDType num_nodes = 0;
  for (IDType u = 0; u < csr.n; u++) {
    if constexpr (USE_LOCAL_BP)
      if (psl_ptr->local_bp->used[u]) continue;

    if constexpr (USE_GLOBAL_BP)
      if (psl_ptr->global_bp->used[u]) continue;

    if constexpr (ELIM_MIN)
      if (psl_ptr->local_min[u]) continue;

    if constexpr (ELIM_LEAF)
      if (psl_ptr->leaf_root[u] != -1) continue;

    if (csr.row_ptr[u] != csr.row_ptr[u + 1]) {
      nodes_to_process[num_nodes++] = u;
    }
  }

  Log("Starting DN Loop");
  bool updated = true;
  last_dist = 1;
  for (int d = 2; d < MAX_DIST; d++) {
    vector<vector<IDType> *> new_labels(csr.n, nullptr);

    size_t ln_labels = 0;
    size_t ln_cut_labels = 0;

    start = omp_get_wtime();
    last_dist = d;
    updated = false;
    Log("Pulling...");
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) reduction(|| : updated) schedule(SCHEDULE)
    for (IDType i = 0; i < num_nodes; i++) {
      int tid = omp_get_thread_num();

      IDType u = nodes_to_process[i];
      /* cout << "Pulling for u=" << u << endl; */

      bool run = true;
      if constexpr (MAX_RANK_PRUNE) run = should_run[u];

      if (run) {
        if constexpr (SMART_DIST_CACHE_CUTOFF) {
          if (psl.labels[u].vertices.size() <= SMART_DIST_CACHE_CUTOFF) {
            new_labels[u] = psl.Pull<false>(u, d, caches[tid], used[tid]);
            // cacheless_pull++;
          } else {
            new_labels[u] = psl.Pull<true>(u, d, caches[tid], used[tid]);
            // cacheful_pull++;
          }
        } else {
          new_labels[u] = psl.Pull<true>(u, d, caches[tid], used[tid]);
        }

        if (new_labels[u] != nullptr && !new_labels[u]->empty()) {
          updated = true;
        }
      }
    }
    end = omp_get_wtime();
    PrintTime("Level " + to_string(d), end - start);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE) reduction(+ : ln_labels)
    for (IDType i = 0; i < num_nodes; i++) {
      IDType u = nodes_to_process[i];
      if (!in_cut[u] && new_labels[u] != nullptr && !new_labels[u]->empty()) {
        auto &labels = psl.labels[u].vertices;
        sort(new_labels[u]->begin(), new_labels[u]->end(), greater<IDType>());
        labels.insert(labels.end(), new_labels[u]->begin(),
                      new_labels[u]->end());
        ln_labels += new_labels[u]->size();
      }
    }

    Barrier();
    Log("Merging Labels for d=" + to_string(d));
    start = omp_get_wtime();
    size_t merge_res = MergeCut(new_labels, psl);
    ln_labels += merge_res;
    ln_cut_labels = merge_res;
    updated = updated || (merge_res != 0);
    end = omp_get_wtime();
    total_merge_time += end - start;
    PrintTime("Merge " + to_string(d), end - start);
    Log("Merging Labels for d=" + to_string(d) + " End");

    cout << "P" << pid << ": " << "Level " << d << " Count: " << ln_labels << " (" << ln_cut_labels << ")" << endl; 
    
    fill(should_run, should_run + csr.n, false);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
    for (IDType u = 0; u < csr.n; u++) {
      auto &labels_u = psl.labels[u];
      labels_u.dist_ptrs.push_back(labels_u.vertices.size());

      if (new_labels[u] != nullptr) {
        delete new_labels[u];
        new_labels[u] = nullptr;
      }


      if constexpr (MAX_RANK_PRUNE){

        IDType start_neighbors = csr.row_ptr[u];
        IDType end_neighbors = csr.row_ptr[u + 1];

        if constexpr (ELIM_MIN)
          if (psl.local_min[u]) {
            for (IDType i = start_neighbors; i < end_neighbors; i++) {
              IDType v = csr.col[i];
              psl.max_ranks[u] = max(psl.max_ranks[u], psl.prev_max_ranks[v]);
            }
          }

        if (psl.max_ranks[u] != -1) {
          for (IDType i = start_neighbors; i < end_neighbors; i++) {
            IDType v = csr.col[i];
            if (v < psl.max_ranks[u]) {
              should_run[v] = true;
            }
          }
        }

        if constexpr (ELIM_MIN) psl.prev_max_ranks[u] = psl.max_ranks[u];
        psl.max_ranks[u] = -1;
      }
    }

    // Stops the execution once all processes agree that they are done
    int updated_int = (int)updated;
    for (int i = 0; i < np; i++) {
      if (pid == i) {
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

  size_t total_label_count = 0;
  for (auto &l : psl_ptr->labels) {
    total_label_count += l.vertices.size();
  }

  size_t label_memory = total_label_count * sizeof(IDType);
  cout << "P" << pid << ":"
       << " Label Memory, " << label_memory / (double)(1024 * 1024 * 1024)
       << " GB" << endl;

  if(pid == 0){
    size_t cut_label_count = 0;
    for(IDType u : cut){
      cut_label_count += psl_ptr->labels[u].vertices.size();
    }

    cout << "Cut Memory: " << (cut_label_count * sizeof(IDType)) / (double) (1024*1024*1024) << " GB" << endl;
  }

#ifdef DEBUG
  Log("Prune by Rank: " + to_string(psl_ptr->prune_rank));
  Log("Prune by Local BP: " + to_string(psl_ptr->prune_local_bp));
  Log("Prune by Global BP: " + to_string(psl_ptr->prune_global_bp));
  Log("Prune by Labels: " + to_string(psl_ptr->prune_labels));
  WriteStats(psl.stats_vec, "stats_p" + to_string(pid) + ".txt");
#endif
}

DPSL::DPSL(int pid, CSR *csr, int np, string partition_str,
           string partition_params)
    : whole_csr(csr), pid(pid), np(np) {
  if (pid == 0) {
    InitP0(partition_str, partition_params);
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

  if (part_csr != nullptr) delete part_csr;

  // if(whole_csr != nullptr)
  //   delete whole_csr;

  delete psl_ptr;

  delete[] partition;

  if (vc_ptr != nullptr) delete vc_ptr;

  if (global_bp != nullptr) delete global_bp;
}
