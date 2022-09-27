#include "psl.h"

void PSL::CountStats(double time){

  long long total_label_count = 0;
  for(IDType u=0; u<csr.n; u++){
    total_label_count += labels[u].vertices.size();
  }
  double avg_label_count = total_label_count / (double) csr.n;

  Stats new_stats;
 
  new_stats.prune_rank = prune_rank; 
  new_stats.prune_labels = prune_labels; 
  new_stats.prune_local_bp = prune_local_bp;
  new_stats.prune_global_bp = prune_global_bp;
  new_stats.total_label_count = total_label_count;
  new_stats.avg_label_count = avg_label_count;
  new_stats.time_elapsed = time;

  stats_vec.push_back(new_stats);
}

PSL::~PSL(){
  if(local_bp != nullptr)
    delete local_bp;

  if(caches != nullptr){
   
   for(int i=0; i<NUM_THREADS; i++){
    delete [] caches[i];
   }
   delete[] caches; 
  }

  if(used != nullptr){
   delete [] used;
  }


}

void PSL::CountPrune(int i){
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


PSL::PSL(CSR& csr_, string order_method, vector<IDType>* cut, BP* global_bp, vector<IDType>* ranks_ptr, vector<IDType>* order_ptr) : csr(csr_),  labels(csr.n), global_bp(global_bp) {

#ifdef DEBUG
  cand_counts.resize(csr.n, 0);
#endif

  if(ranks_ptr == nullptr){
    order = gen_order<IDType>(csr.row_ptr, csr.col, csr.n, csr.m, order_method);
    ranks.resize(csr.n);
    for(IDType i=0; i<csr.n; i++){
          ranks[order[i]] = i;
    } 
  } else {
    ranks.insert(ranks.end(), ranks_ptr->begin(), ranks_ptr->end());
    order.insert(order.end(), order_ptr->begin(), order_ptr->end());
  }

  in_cut.resize(csr.n, false);
  if(cut != nullptr && !cut->empty()){
    for(IDType u : *cut){
      in_cut[u] = true;
    }
  }

  if(cut == nullptr)
    csr.Reorder(order, nullptr, nullptr);
  
  if constexpr(USE_LOCAL_BP){
    if constexpr(USE_GLOBAL_BP){
      local_bp = new BP(csr, ranks, order, cut, LOCAL_BP_MODE);
    } else {
      local_bp = new BP(csr, ranks, order, nullptr, LOCAL_BP_MODE);
    }
  }

  max_ranks.resize(csr.n, -1);
}


void PSL::WriteLabelCounts(string filename){
  ofstream ofs(filename);
  ofs << "L:\t";
  for(int i=0; i<last_dist; i++){
    ofs << i << "\t";
  }
  ofs << endl;

  long long total = 0;
  long long max_label_size = 0;
  for(IDType u=0; u<csr.n; u++){
    ofs << u << ":\t";
    auto& labels_u = labels[u];
    total += labels_u.vertices.size();

    max_label_size = max((size_t) max_label_size, labels_u.vertices.size());

    for(int d=0; d<last_dist; d++){
      IDType dist_start = labels_u.dist_ptrs[d];
      IDType dist_end = labels_u.dist_ptrs[d+1];

      ofs << dist_end - dist_start << "\t";
    }
    ofs << endl;
  }
  ofs << endl;

  ofs << "Total Label Count: " << total << endl;
  cout << "Total Label Count: " << total << endl;
  ofs << "Avg. Label Count: " << total/(double) csr.n << endl;
  cout << "Avg. Label Count: " << total/(double) csr.n << endl;
  cout << "Max Label Count: " << max_label_size << endl;

  ofs << endl;

  ofs.close();

#ifdef DEBUG
ofstream ofs2("output_psl_cand_counts.txt");
for(size_t i=0; i < cand_counts.size(); i++){
  ofs2 << i << ": " << cand_counts[i] << endl;
}
ofs2.close();
#endif

}


void PSL::QueryTest(int query_count){

  vector<IDType> sources;
  sources.reserve(query_count);
  for(int i=0; i<query_count; i++){
    sources.push_back((IDType) random_range(0,csr.n));
  }

  for(IDType u: sources){
    cout << "Query From: " << u << endl;
    double start_time = omp_get_wtime();
    auto results = Query(u);
    double end_time = omp_get_wtime();

    // cout << "Avg. Query Time: " << (end_time-start_time) / csr.n << " seconds" << endl;

    start_time = omp_get_wtime();
    auto bfs_results = BFSQuery(csr, u);
    end_time = omp_get_wtime();
    
    // cout << "Avg. BFS Time: " << (end_time-start_time) / csr.n << " seconds" << endl;

    bool all_correct = true;
    for(IDType i=0; i<csr.n; i++){
      int psl_res = results->at(i);
      int bfs_res = bfs_results->at(i);
      string correctness = (bfs_res == psl_res) ? "correct" : "wrong";

      if(bfs_res != psl_res){
        all_correct = false;
      }

    }

    cout << "Correctness: " << all_correct << endl;

    delete results;
    delete bfs_results;
  }

}

void PSL::Query(IDType u, string filename){
  double start_time = omp_get_wtime();
  auto results = Query(u);
  double end_time = omp_get_wtime();

  cout << "Avg. Query Time: " << (end_time-start_time) / csr.n << " seconds" << endl;

  start_time = omp_get_wtime();
  auto bfs_results = BFSQuery(csr, u);
  end_time = omp_get_wtime();
  
  cout << "Avg. BFS Time: " << (end_time-start_time) / csr.n << " seconds" << endl;
  

#ifdef DEBUG
  ofstream ofs(filename);
  ofs << "Source: " << u << endl;
  ofs << "Target\tPSL_Distance\tBFS_Distance\tCorrectness" << endl;
#endif

  bool all_correct = true;
  for(IDType i=0; i<csr.n; i++){
    int psl_res = results->at(i);
    int bfs_res = bfs_results->at(i);
    string correctness = (bfs_res == psl_res) ? "correct" : "wrong";

    if(bfs_res != psl_res){
      all_correct = false;
    }

#ifdef DEBUG
    ofs << i << "\t" << psl_res << "\t" << bfs_res << "\t" << correctness << endl;
#endif
  }

  cout << "Correctness: " << all_correct << endl;

#ifdef DEBUG
  ofs.close();
#endif

  delete results;
  delete bfs_results;
}

vector<IDType>* PSL::Query(IDType u) {


  vector<IDType>* results = new vector<IDType>(csr.n, MAX_DIST);

  auto &labels_u = labels[u];

  vector<char> cache(csr.n, -1);

  for (int d = 0; d < last_dist; d++) {
    IDType dist_start = labels_u.dist_ptrs[d];
    IDType dist_end = labels_u.dist_ptrs[d + 1];

    for (IDType i = dist_start; i < dist_end; i++) {
      IDType w = labels_u.vertices[i]; 
      cache[w] = (char) d;
    }
  }

  for (IDType v = 0; v < csr.n; v++) {

    auto& labels_v = labels[v];

    int min = MAX_DIST;

    if constexpr(USE_LOCAL_BP)
      min = local_bp->QueryByBp(u,v);

    for (int d = 0; d < min && d < last_dist; d++) {
      IDType dist_start = labels_v.dist_ptrs[d];
      IDType dist_end = labels_v.dist_ptrs[d + 1];

      for (IDType i = dist_start; i < dist_end; i++) {
        IDType w = labels_v.vertices[i];

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

template<bool use_cache = true>
bool PSL::Prune(IDType u, IDType v, int d, char* cache) {
  auto &labels_v = labels[v];
  auto &labels_u = labels[u];

  for (int i = 0; i < d; i++) {
    IDType dist_start = labels_v.dist_ptrs[i];
    IDType dist_end = labels_v.dist_ptrs[i + 1];

    for (IDType j = dist_start; j < dist_end; j++) {
      IDType x = labels_v.vertices[j];
      
      //TODO: Absolute value with negative marks ?

      if constexpr(use_cache){
        int cache_dist = cache[x];

        if ((i + cache_dist) <= d) {
          return true;
        }

      } else {
        int desired_dist = (i == 0) ? d : d - i + 1;
        IDType dist_start2 = labels_u.dist_ptrs[0];
        IDType dist_end2 = labels_u.dist_ptrs[desired_dist];
    
        for (IDType j2 = dist_start2; j2 < dist_end2; j2++) {
          IDType x2 = labels_u.vertices[j2];

          if(x2 == x){
            return true;
          }  
        }
      }
    }
  }

  return false;
}

template<bool use_cache = true>
vector<IDType>* PSL::Pull(IDType u, int d, char* cache, vector<bool>& used_vec) {

  if constexpr(USE_LOCAL_BP)
    if(local_bp->used[u])
      return nullptr;
    

  if constexpr(USE_GLOBAL_BP)
    if(global_bp->used[u])
      return nullptr;

  IDType start = csr.row_ptr[u];
  IDType end = csr.row_ptr[u + 1];
  
  if(end == start){
    return nullptr;
  }

  auto &labels_u = labels[u];

  vector<IDType>* new_labels = nullptr;
  vector<IDType> candidates;

  for (IDType i = start; i < end; i++) {
    IDType v = csr.col[i];
    auto &labels_v = labels[v];

    IDType labels_start = labels_v.dist_ptrs[d-1];
    IDType labels_end = labels_v.dist_ptrs[d];

    for (IDType j = labels_start; j < labels_end; j++) {
      IDType w = labels_v.vertices[j];

      if (u > w) {
	      CountPrune(PRUNE_RANK);
        continue;
      }

      if(used_vec[w]){
        continue;
      }

      if constexpr(USE_GLOBAL_BP){
        if(global_bp->PruneByBp(u, w, d)){
          CountPrune(PRUNE_GLOBAL_BP);
          continue;
        }
      }

      if constexpr(USE_LOCAL_BP){
        if(local_bp->PruneByBp(u, w, d)){
          CountPrune(PRUNE_LOCAL_BP);
          continue;
        }
      }

      used_vec[w] = true;
      candidates.push_back(w);
      
    }
  }

  if(candidates.empty()){
    return nullptr;
  }

#ifdef DEBUG
  for(IDType w : candidates){
    cand_counts[w]++;
  }
#endif


  if constexpr(use_cache)
    for (int i = 0; i < d; i++) {
      IDType dist_start = labels_u.dist_ptrs[i];
      IDType dist_end = labels_u.dist_ptrs[i + 1];

      for (IDType j = dist_start; j < dist_end; j++) {
        IDType w = labels_u.vertices[j];
        cache[w] = (char) i;
      }
    }

  for(IDType w : candidates){

    used_vec[w] = false;

    if constexpr(use_cache)
      if(cache[w] <= d){
        continue;
      }

    if(Prune<use_cache>(u, w, d, cache)){
      CountPrune(PRUNE_LABEL);
      continue;
    }

    if(new_labels == nullptr){
      new_labels = new vector<IDType>;
    }

    new_labels->push_back(w);

    if(w > max_ranks[u]){
      max_ranks[u] = w;
    }

  }


  if constexpr(use_cache) 
  {
    IDType dist_start = labels_u.dist_ptrs[0];
    IDType dist_end = labels_u.dist_ptrs[d];

    for (IDType j = dist_start; j < dist_end; j++) {
      IDType w = labels_u.vertices[j];
      cache[w] = (char) MAX_DIST;
    }
  }

  
  return new_labels;
}

vector<IDType>* PSL::Init(IDType u){
  
  max_ranks[u] = u;

  vector<IDType>* init_labels = new vector<IDType>;

  IDType start = csr.row_ptr[u];
  IDType end = csr.row_ptr[u + 1];

  for (IDType j = start; j < end; j++) {
    IDType v = csr.col[j];

    if (v > u) {
      if constexpr(USE_LOCAL_BP)
        if(local_bp->used[v]){
          continue;
        }

      if constexpr(USE_GLOBAL_BP)
        if(global_bp->used[v]){
          continue;
        }

      init_labels->push_back(v);

      if(v > max_ranks[u])
        max_ranks[u] = v;
    }
  }

  return init_labels;

}


void PSL::Index() {


  double start_time, end_time, all_start_time, all_end_time;
  all_start_time = omp_get_wtime();

  caches = new char*[NUM_THREADS];
  #pragma omp parallel for default(shared) num_threads(NUM_THREADS)
  for(int i=0; i<NUM_THREADS; i++){
    caches[i] = new char[csr.n];
    fill(caches[i], caches[i] + csr.n, MAX_DIST);
  }

  used = new vector<bool>[NUM_THREADS];
  #pragma omp parallel for default(shared) num_threads(NUM_THREADS)
  for(int i=0; i<NUM_THREADS; i++){
    used[i].resize(csr.n, false);
  }

  bool should_run[csr.n];
  fill(should_run, should_run+csr.n, true);

  // Adds the first two level of vertices
  // Level 0: vertex to itself
  // Level 1: vertex to neighbors
  long long l1_count = 0;
  long long l0_count = 0;
  start_time = omp_get_wtime();
  #pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE) reduction(+ : l0_count, l1_count)
  for (IDType u = 0; u < csr.n; u++) {

    bool should_init = true;
    if constexpr(USE_LOCAL_BP)
      if(local_bp->used[u])
        should_init = false;
     
     if constexpr(USE_GLOBAL_BP)
      if(global_bp->used[u])
        should_init = false;

    if(should_init){
      labels[u].vertices.push_back(u);
      l0_count++;
      auto init_labels = Init(u);
      // sort(init_labels.begin(), init_labels.end())
      labels[u].vertices.insert(labels[u].vertices.end(), init_labels->begin(), init_labels->end());
      l1_count += init_labels->size();
      delete init_labels;

      labels[u].dist_ptrs.push_back(0);
      labels[u].dist_ptrs.push_back(1);
      labels[u].dist_ptrs.push_back(labels[u].vertices.size());

      IDType ngh_start = csr.row_ptr[u];
      IDType ngh_end = csr.row_ptr[u+1];

      for(IDType i=ngh_start; i<ngh_end; i++){
        IDType v = csr.col[i];

        if(v < max_ranks[u])
          should_run[v] = true;
      }

      max_ranks[u] = -1;

    } else {
      labels[u].dist_ptrs.push_back(0);
      labels[u].dist_ptrs.push_back(0);
      labels[u].dist_ptrs.push_back(0);
    }
   
  }
  end_time = omp_get_wtime();
  cout << "Level 0 & 1 Time: " << end_time-start_time << " seconds" << endl;
  cout << "Level 0 & 1 Count: " << l0_count << "," << l1_count << endl;

  // long long cacheless_pull = 0;
  // long long cacheful_pull = 0;

  vector<vector<IDType>*> new_labels(csr.n, nullptr);
  bool updated = true;

  for (int d = 2; d < MAX_DIST && updated; d++) {
   
 
    start_time = omp_get_wtime();
    updated = false;


    // TODO: Reverse this loop
    #pragma omp parallel for default(shared) num_threads(NUM_THREADS) reduction(||:updated) schedule(SCHEDULE)
    for (IDType u = 0; u < csr.n; u++) {
      int tid = omp_get_thread_num();

      if(should_run[u]){

        if constexpr(SMART_DIST_CACHE_CUTOFF)
        {
          if(labels[u].vertices.size() <= SMART_DIST_CACHE_CUTOFF){
            new_labels[u] =  Pull<false>(u, d, caches[tid], used[tid]);
            // cacheless_pull++;
          }
          else {
            new_labels[u] =  Pull<true>(u, d, caches[tid], used[tid]);
            // cacheful_pull++;
          }
        } else {
          new_labels[u] =  Pull<true>(u, d, caches[tid], used[tid]);
        }


        updated = updated || (new_labels[u] != nullptr && !new_labels[u]->empty());
        should_run[u] = false;
      }
    }   


    last_dist++;

    
    long long level_count = 0;      
    #pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE) reduction(+ : level_count)
    for (IDType u = 0; u < csr.n; u++) {
      
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
      level_count += (labels_u.vertices.size() - labels_u.dist_ptrs[d]);
      delete new_labels[u];
      new_labels[u] = nullptr;

      IDType start = csr.row_ptr[u]; 
      IDType end = csr.row_ptr[u+1];

      for(IDType i=start; i<end; i++){
        IDType v = csr.col[i];
        if(v < max_ranks[u]){
          should_run[v] = true;
        }
      } 

      max_ranks[u] = -1;
    }

#ifdef DEBUG
    CountStats(omp_get_wtime() - all_start_time);
#endif

    end_time = omp_get_wtime();
    cout << "Level " << d << " Time: " << end_time-start_time << " seconds" << endl;
    cout << "Level " << d << " Count: " << level_count << endl;
  }

  all_end_time = omp_get_wtime();
  cout << "Indexing: " << all_end_time-all_start_time << " seconds" << endl;

  size_t total_label_count = 0;
  for(auto& l : labels){
    total_label_count += l.vertices.size();
  }
  size_t label_memory = total_label_count * sizeof(IDType);
  cout << "Label Memory: " << label_memory / (double) (1024*1024*1024) << " GB" << endl;



  // cout << "Cacheless: " << cacheless_pull << endl; 
  // cout << "Cacheful: " << cacheful_pull << endl; 

#ifdef DEBUG
  cout << "Prune by Rank: " << prune_rank << endl; 
  cout << "Prune by Local BP: " << prune_local_bp << endl; 
  cout << "Prune by Global BP: " << prune_global_bp << endl; 
  cout << "Prune by Labels: " << prune_labels << endl; 
  WriteStats(stats_vec, "stats.txt");
#endif
}

template vector<IDType>* PSL::Pull<true>(IDType u, int d, char* cache, vector<bool>& used_vec);
template vector<IDType>* PSL::Pull<false>(IDType u, int d, char* cache, vector<bool>& used_vec);