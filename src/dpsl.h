#ifndef DPSL_H
#define DPSL_H

#include "common.h"
#include "external/toml/toml.h"
#include "psl.h"
#include "mpi.h"
#include <algorithm>
#include <fstream>
#include <ostream>
#include <string>
#include <unordered_set>
#include "cut.h"

using namespace std;

enum MPI_CONSTS{
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
};


class DPSL{

public:

    template <typename T>
    void SendData(T* data, int size, int vertex, int to, MPI_Datatype type=MPI_INT32_T);

    template <typename T>
    void BroadcastData(T* data, int size, int vertex, MPI_Datatype type=MPI_INT32_T);

    template <typename T>
    int RecvData(T*& data,int vertex, int from, MPI_Datatype type=MPI_INT32_T);


    bool MergeCut(vector<vector<int>*> new_labels, PSL& psl, bool init=false);
    void Barrier();
    int pid, np, global_n;
    CSR* whole_csr;
    CSR* part_csr;
    BP* global_bp;
    int* partition;
    vector<int> cut;
    vector<int> names;
    VertexCut* vc_ptr;
    int last_dist;
    const toml::Value& config;
    char** caches;
    void InitP0();
    void Init();
    void Index();
    void WriteLabelCounts(string filename);
    void Query(int u, string filename);
    void Log(string msg);
    void PrintTime(string tag, double time);
    PSL * psl_ptr;
    DPSL(int pid, CSR* csr, const toml::Value& config, int np);
};

inline void DPSL::Log(string msg){
#ifdef DEBUG
  cout << "P" << pid << ": " << msg << endl;
#endif
}

inline void DPSL::PrintTime(string tag, double time){
  cout << "P" << pid << ": " << tag << ", " << time << " seconds" << endl;
}


inline void DPSL::Query(int u, string filename){
  
  Log("Starting Query");
  Log("Global N: " + to_string(global_n));
  Barrier();
 
  int* vertices_u;
  int* dist_ptrs_u;
  int dist_ptrs_u_size;
  int vertices_u_size;

  if(partition[u] == pid){
      Log("Broadcasting u's labels");
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


  Log("Constructing cache");
  int cache[global_n];
  fill(cache, cache+global_n, -1);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
  for(int d=0; d<dist_ptrs_u_size-1; d++){
      int start = dist_ptrs_u[d];
      int end = dist_ptrs_u[d+1];
  
      for(int i= start; i<end; i++){
            int v = vertices_u[i];
            cache[v] = d;
        }
    }

  Log("Querying locally");
  vector<int> local_dist(part_csr->n);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) 
  for(int v=0; v<part_csr->n; v++){
    
    int min = MAX_DIST;
    if constexpr(USE_GLOBAL_BP){
      min = global_bp->QueryByBp(u, v);
    }

    if constexpr(USE_LOCAL_BP){
      int local_bp_dist = psl_ptr->local_bp->QueryByBp(u, v);
      if(local_bp_dist < min){
        min = local_bp_dist;
      }
    }

    auto& vertices_v = psl_ptr->labels[v].vertices;
    auto& dist_ptrs_v = psl_ptr->labels[v].dist_ptrs;
   
    for(int d=0; d<last_dist && d < min; d++){
      int start = dist_ptrs_v[d];
      int end = dist_ptrs_v[d+1];

      for(int i= start; i<end; i++){
        int w = vertices_v[i];
       
        if(cache[w] == -1){
          continue;
        }

        int dist = d + cache[w];        
        if(dist < min){
          min = dist;
        }
      }
    }

    local_dist[v] = min;
  }

  Barrier();
  
  Log("Synchronizing query results");
  if(pid == 0){
    int all_dists[whole_csr->n];
    int source[whole_csr->n];
    fill(all_dists, all_dists+whole_csr->n, -1);
    fill(source, source+whole_csr->n, -1);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for(int i=0; i<local_dist.size(); i++){
      all_dists[i] = local_dist[i];
      source[i] = 0;
    }

    for(int p=1; p<np; p++){
      int* dists;
      int size = RecvData(dists, 0, p);
      for(int i=0; i<size; i++){
        if(all_dists[i] == -1 || all_dists[i] > dists[i]){
          all_dists[i] = dists[i];
          source[i] = p;
        }
      }
      delete [] dists;
    }

    vector<int>* bfs_results = BFSQuery(*whole_csr, u);

    ofstream ofs(filename);
    ofs << "Vertex\tDPSL(source)\tBFS\tCorrectness" << endl;
    for(int i=0; i<whole_csr->n; i++){
      int psl_res = all_dists[i];
      if(psl_res == MAX_DIST){
        psl_res = -1;
      }
      int bfs_res = (*bfs_results)[i];
      string correctness = (bfs_res == psl_res) ? "correct" : "wrong";
      ofs << i << "\t" << psl_res << "(" << source[i] << ")" << "\t" << bfs_res << "\t" << correctness << endl;
    }
    delete bfs_results;
    ofs.close();

  } else {
    SendData(local_dist.data(), local_dist.size(), 0, 0);
  }
}

inline void DPSL::WriteLabelCounts(string filename){
  
  Barrier();
  CSR& csr = *part_csr;

  int counts[part_csr->n];
  fill(counts, counts + part_csr->n, -1);

  for(int i=0; i<part_csr->n; i++){
    counts[i] = psl_ptr->labels[i].vertices.size();
  }

  if(pid != 0)
    SendData(counts, part_csr->n, 0, 0);

  
  if(pid == 0){
    
    int source[whole_csr->n]; 
    int all_counts[whole_csr->n];
    fill(all_counts, all_counts + whole_csr->n, -1);
    fill(source, source + whole_csr->n, -1); // -1 indicates free floating vertex

    for(int i=0; i<part_csr->n; i++){
      all_counts[i] = counts[i];
      source[i] = 0;  // 0 indicates cut vertex as well as partition 0
    }

    for(int p=1; p<np; p++){
      int* recv_counts;
      int size = RecvData(recv_counts, 0, p);
      for(int i=0; i<size; i++){
        if(recv_counts[i] != -1){ // Count recieved
          if(vc_ptr->cut.find(i) == vc_ptr->cut.end() && all_counts[i] < recv_counts[i]){  // vertex not in cut and counted on the recieved data
            
            all_counts[i] = recv_counts[i]; // Update count
            source[i] = p;

          } 
        }
      } 
    }


    int total_per_source[np];
    fill(total_per_source, total_per_source+np, 0);
    for(int i=0; i<part_csr->n; i++){
      total_per_source[source[i]]++; 
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
    
    for(int p=0; p<np; p++){
      ofs << "Total for P" << p << ": " << total_per_source[p] << endl;
    }
    
    ofs << endl;

    ofs.close();
  }
}

inline bool DPSL::MergeCut(vector<vector<int>*> new_labels, PSL& psl, bool init){
  
  bool updated = false;

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

      if(merged_labels.size() > 0){
        Log("Merging Labels for " + to_string(i));

        if(init){
          merged_labels.erase(u);
          labels_u.push_back(u);
        }

        labels_u.insert(labels_u.end(), merged_labels.begin(), merged_labels.end()); 
      }
      
      Log("Broadcasting Labels for " + to_string(i));
      if(labels_u.size() > start){
        updated = true;
        BroadcastData(labels_u.data() + start, labels_u.size()-start, i);
      } else {
        BroadcastData<int>(nullptr, 0, i);
      }

    }
  } else {

    for(int i=0; i<cut.size(); i++){
      int u = cut[i];
      Log("Iteration u=" + to_string(u));
      auto& labels_u = psl.labels[u].vertices;
      int new_labels_size = (new_labels[u] == nullptr) ? 0 : new_labels[u]->size();
      int* new_labels_data = (new_labels[u] == nullptr) ? nullptr : new_labels[u]->data();
      Log("Sending Labels for " + to_string(i) + " with size " + to_string(new_labels_size));
      SendData(new_labels_data, new_labels_size, i, 0);
      int* merged_labels;
      Log("Recieving Labels for " + to_string(i));
      int size = RecvData(merged_labels, i, 0);
      Log("Recieving Labels for " + to_string(i) + " End");
      
      if(size > 0 && merged_labels != nullptr){
        updated = true;
        labels_u.insert(labels_u.end(), merged_labels, merged_labels + size);
        delete[] merged_labels;
      }
      Log("Inserting Labels for " + to_string(i) + " End");
    }
  }

  return updated;
}

inline void DPSL::Barrier(){
  MPI_Barrier(MPI_COMM_WORLD);
}

template <typename T>
inline void DPSL::SendData(T* data, int size, int vertex, int to, MPI_Datatype type){
  int tag = (vertex << 1);
  int size_tag = tag | 1;

  MPI_Send(&size, 1, MPI_INT32_T, to, size_tag, MPI_COMM_WORLD);

  if(size != 0 && data != nullptr)
    MPI_Send(data, size, type, to, tag, MPI_COMM_WORLD);
}

// TODO: Replace this with MPI_Bcast
template <typename T>
inline void DPSL::BroadcastData(T *data, int size, int vertex, MPI_Datatype type){
  for(int p=0; p<np; p++){
    if(p != pid)
      SendData(data, size, vertex, p, type);
  }
}

template <typename T>
inline int DPSL::RecvData(T *& data, int vertex, int from, MPI_Datatype type){
    int tag = (vertex << 1);
    int size_tag = tag | 1;
    int size = 0;

    int error_code1, error_code2;

    error_code1 = MPI_Recv(&size, 1, MPI_INT32_T, from, size_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if(size != 0){
      data = new T[size];
      error_code2 = MPI_Recv(data, size, type, from, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    global_n = csr.n;

    vc_ptr = new VertexCut(csr, order_method, np, config);
    
    VertexCut& vc = *vc_ptr;

    
    Log("Creating All Cut");
    cut.insert(cut.end(), vc.cut.begin(), vc.cut.end());
    auto& ranks = vc.ranks;

    Log("Ordering Cut By Rank");
    sort(cut.begin(), cut.end(), [ranks](int u, int v){
        return ranks[u] > ranks[v];
    });

    auto& csrs = vc.csrs;
    part_csr = csrs[0];

    Log("Initial Barrier Region");
    Barrier();
    for(int i=1; i<np; i++){
        SendData(&global_n, 1, MPI_GLOBAL_N, i);
        SendData(csrs[i]->row_ptr, (csrs[i]->n)+1, MPI_CSR_ROW_PTR, i);
        SendData(csrs[i]->col, csrs[i]->m, MPI_CSR_COL, i);
        SendData(vc.partition, csr.n, MPI_PARTITION, i);
        SendData(cut.data(), cut.size(), MPI_CUT, i);
    }

    partition = vc.partition;
    
    Barrier();
    Log("Initial Barrier Region End");

    if constexpr(USE_GLOBAL_BP){
      Log("Creating Global BP");
      global_bp = new BP(csr, vc.ranks, vc.order, &cut);

      Log("Global BP Barrier Region");
      Barrier();
      for(int i=0; i<global_n; i++){
        BPLabel& bp_label = global_bp->bp_labels[i];
        BroadcastData(bp_label.bp_dists, N_ROOTS, MPI_BP_DIST, MPI_UINT8_T);
        vector<uint64_t> bp_sets;
        bp_sets.reserve(N_ROOTS*2);
        for(int j=0; j<N_ROOTS; j++){
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

    caches = new char*[NUM_THREADS];
    for(int i=0; i<NUM_THREADS; i++){
      caches[i] = new char[part_csr->n];
    }

}

inline void DPSL::Init(){
    int *row_ptr;
    int *col;
    int *cut_ptr;

    Log("Initial Barrier Region");
    Barrier();
    
    int* global_n_ptr;
    RecvData(global_n_ptr, MPI_GLOBAL_N, 0);
    global_n = *global_n_ptr;
    delete global_n_ptr;


    int size_row_ptr = RecvData(row_ptr, MPI_CSR_ROW_PTR, 0);
    int size_col = RecvData(col, MPI_CSR_COL, 0);
    int size_partition = RecvData(partition, MPI_PARTITION, 0);
    int size_cut = RecvData(cut_ptr, MPI_CUT, 0);
    Barrier();
    Log("Initial Barrier Region End");

    if constexpr(USE_GLOBAL_BP){
      Log("Global BP Barrier Region");
      Barrier();
      vector<BPLabel> bp_labels(global_n);
      for(int i=0; i<global_n; i++){
        uint8_t* bp_dists;
        RecvData(bp_dists, MPI_BP_DIST, 0, MPI_UINT8_T);
        copy(bp_dists, bp_dists + N_ROOTS, bp_labels[i].bp_dists);
        Log("Recieved dists");
        
        uint64_t* bp_sets;
        RecvData(bp_sets, MPI_BP_SET, 0, MPI_UINT64_T);
        Log("Recieved sets");

        for(int j=0; j<N_ROOTS; j++){
          bp_labels[i].bp_sets[j][0] = bp_sets[j*2]; 
          bp_labels[i].bp_sets[j][1] = bp_sets[j*2+1]; 
        }

        delete[] bp_dists;
        delete[] bp_sets;
        Barrier();
      }
      Barrier();
      Log("Global BP Barrier Region End");

      global_bp = new BP(bp_labels);
    }

    cut.insert(cut.end(), cut_ptr, cut_ptr+size_cut);

    part_csr = new CSR(row_ptr, col, size_row_ptr-1, size_col);
    Log("CSR Dims: " + to_string(part_csr->n) + "," + to_string(part_csr->m));
    delete[] cut_ptr;

    caches = new char*[NUM_THREADS];
    for(int i=0; i<NUM_THREADS; i++){
      caches[i] = new char[part_csr->n];
    }
}

inline void DPSL::Index(){
 
    double start, end, alg_start, alg_end;
    Log("Indexing Start");
    CSR& csr = *part_csr;

    string order_method = config.find("order_method")->as<string>();
    psl_ptr = new PSL(*part_csr, order_method, &cut, global_bp);
    PSL& psl = *psl_ptr;

    start = omp_get_wtime();
    alg_start = omp_get_wtime();
    vector<vector<int>*> init_labels(csr.n, nullptr);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for(int u=0; u<csr.n; u++){
      init_labels[u] = psl.Init(u);
    }
    
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) 
    for(int u=0; u<csr.n; u++){
      if(psl.ranks[u] < psl.min_cut_rank && init_labels[u] != nullptr && !init_labels[u]->empty()){
        auto& labels = psl.labels[u].vertices;
        labels.insert(labels.end(), init_labels[u]->begin(), init_labels[u]->end());
      }
    }
    end = omp_get_wtime();
    PrintTime("Level 0&1", end-start);

    Log("Merging Initial Labels");
    start = omp_get_wtime();
    MergeCut(init_labels, psl, true);
    end = omp_get_wtime();
    PrintTime("Merge 0&1", end-start);
    Log("Merging Initial Labels End");
    
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) 
    for (int u = 0; u < csr.n; u++) {
      auto& labels = psl.labels[u];
      labels.dist_ptrs.reserve(3);
      labels.dist_ptrs.push_back(0);
      labels.dist_ptrs.push_back(1);
      labels.dist_ptrs.push_back(labels.vertices.size());
      delete init_labels[u];
    }

    Barrier();

    Log("Starting DN Loop");
    bool updated = true;
    last_dist = 1;
    for(int d=2; d < MAX_DIST; d++){    

        Barrier();
        vector<vector<int>*> new_labels(csr.n, nullptr);

        start = omp_get_wtime();
        if(updated){
          last_dist = d;
          updated = false;
          Log("Pulling...");
#pragma omp parallel default(shared) num_threads(NUM_THREADS) reduction(||:updated) 
        {
          int tid = omp_get_thread_num();
          int nt = omp_get_num_threads();
          for(int u=tid; u<csr.n; u+=nt){
              new_labels[u] = psl.Pull(u,d,caches[tid]);
              if(psl.ranks[u] < psl.min_cut_rank && new_labels[u] != nullptr && !new_labels[u]->empty()){
                updated = updated || true;
                auto& labels = psl.labels[u].vertices;
                labels.insert(labels.end(), new_labels[u]->begin(), new_labels[u]->end());
              }
            }       
          }
        }
        end = omp_get_wtime();
        PrintTime("Level " + to_string(d), end-start);

        
        Barrier();
        
        Log("Merging Labels for d=" + to_string(d));
        start = omp_get_wtime();
        bool merge_res = MergeCut(new_labels, psl);
        updated = updated || merge_res;
        end = omp_get_wtime();
        PrintTime("Merge " + to_string(d), end-start);
        Log("Merging Labels for d=" + to_string(d) + " End");


#pragma omp parallel for default(shared) num_threads(NUM_THREADS) 
        for(int u=0; u<csr.n; u++){
          auto& labels_u = psl.labels[u];
          labels_u.dist_ptrs.push_back(labels_u.vertices.size());

          if(new_labels[u] != nullptr)
            delete new_labels[u];
        }

        // Stops the execution once all processes agree that they are done
        int updated_int = (int) updated;
        BroadcastData(&updated_int, 1, MPI_UPDATED);
        for(int i=0; i<np-1; i++){
          int* updated_other;
          RecvData(updated_other, MPI_UPDATED, MPI_ANY_SOURCE);
          updated_int |= *updated_other;
          delete updated_other;
        }

        if(updated_int == 0){
          break;
        }
    }

    alg_end = omp_get_wtime();
    PrintTime("Total", alg_end-alg_start);

    Log("Prune by Rank: " + to_string(psl_ptr->prune_rank)); 
    Log("Prune by Local BP: " + to_string(psl_ptr->prune_local_bp)); 
    Log("Prune by Global BP: " + to_string(psl_ptr->prune_global_bp)); 
    Log("Prune by Labels: " + to_string(psl_ptr->prune_labels)); 
}

inline DPSL::DPSL(int pid, CSR* csr, const toml::Value& config, int np): whole_csr(csr), pid(pid), config(config), np(np){
    if(pid == 0){
        InitP0();
    } else {
        Init();
    }

}

#endif
