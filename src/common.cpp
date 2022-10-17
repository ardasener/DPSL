#include "common.h"
#include <random>
#include <time.h>

size_t random_range(const size_t & min, const size_t & max) {
 static mt19937 generator(time(0));
 uniform_int_distribution<size_t> distribution(min, max);
 return distribution(generator);
}

void WriteStats(const vector<Stats>& stats_vec, string filename){
  ofstream ofs(filename);
  ofs << "prune_rank" << "|" << "prune_labels" << "|" << "prune_global_bp" << "|" << "prune_local_bp" << "|" << "total_label_count" << "|" << "avg_label_count" << "|" << "time_elapsed" << endl;
  for(int i=0; i<stats_vec.size(); i++){
    auto& stats = stats_vec[i];
    ofs << "STATS_Level_" << i + 2 << ",";
    ofs << stats.prune_rank << "," << stats.prune_labels << "," << stats.prune_global_bp << "," << stats.prune_local_bp << "," << stats.total_label_count << "," << stats.avg_label_count << "," << stats.time_elapsed << endl;
  }
}

vector<int>* BFSQuery(CSR& csr, IDType u){

  vector<int>* dists = new vector<int>(csr.n, -1);
  auto& dist = *dists;

  IDType* q = new IDType[csr.n];

  IDType q_start = 0;
  IDType q_end = 1;
  q[q_start] = u;

  dist[u] = 0;
  while(q_start < q_end){
    IDType curr = q[q_start++];

    IDType start = csr.row_ptr[curr];
    IDType end = csr.row_ptr[curr+1];

    for(IDType i=start; i<end; i++){
      IDType v = csr.col[i];

      if(dist[v] == -1){
          dist[v] = dist[curr]+1;
          q[q_end++] = v;
      }
    }

  }

  delete[] q;

  return dists;
}

CSR::~CSR(){
    delete[] row_ptr;
    delete[] col;
    
    if(inv_ids != nullptr)
      delete[] inv_ids;

    if(ids != nullptr)
      delete[] ids;
}

  CSR::CSR(CSR& csr){
    row_ptr = new IDType[csr.n+1];
    col = new IDType[csr.m];
    ids = new IDType[csr.n];
    inv_ids = new IDType[csr.n];
    type = new char[csr.n];

#pragma omp parallel for num_threads(NUM_THREADS)
    for(IDType i=0; i<csr.n+1; i++){
      row_ptr[i] = csr.row_ptr[i];
    }

#pragma omp parallel for num_threads(NUM_THREADS)
    for(IDType i=0; i<csr.m; i++){
      col[i] = csr.col[i];
    }

#pragma omp parallel for num_threads(NUM_THREADS)
    for(IDType i=0; i<csr.n; i++){
      ids[i] = csr.ids[i];
      inv_ids[i] = csr.inv_ids[i];
      type[i] = csr.type[i];
    }

    n = csr.n;
    m = csr.m;

    Sort();
    InitIds();

  }

  CSR::CSR(IDType * row_ptr, IDType *col, IDType* ids, IDType* inv_ids, char* type, IDType n, IDType m): 
    row_ptr(row_ptr), col(col), ids(ids), inv_ids(inv_ids), type(type), n(n), m(m) {
    }

  CSR::CSR(string filename) {

    bool is_mtx = false;
    if(filename.find(".mtx") != string::npos){
      is_mtx = true;
    }

    PigoCOO pigo_coo(filename);

    IDType *coo_row = pigo_coo.x();
    IDType *coo_col = pigo_coo.y();
    m = pigo_coo.m();
    n = pigo_coo.n();
    cout << "N:" << n << endl;
    cout << "M:" << m << endl;

    if(is_mtx){
      cout << "Changing mtx indices to zero-based" << endl;
#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
      for(IDType i=0; i<m; i++){
        coo_row[i]--;
        coo_col[i]--;
      }
    }


    vector<pair<IDType,IDType>> edges(m);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS)
    for(IDType i=0; i<m; i++){
      edges[i] = make_pair(coo_row[i], coo_col[i]);
    }

    sort(edges.begin(), edges.end(), less<pair<IDType, IDType>>());

    auto unique_it = unique(edges.begin(), edges.end(), [](const pair<IDType,IDType>& p1, const pair<IDType,IDType>& p2){
	    return (p1.first == p2.first) && (p1.second == p2.second);
	  });

    edges.erase(unique_it, edges.end());

    m = edges.size();
    cout << "Unique M:" << m << endl;

    row_ptr = new IDType[n + 1];
    col = new IDType[m];

    fill(row_ptr, row_ptr+n+1, 0);

    for (IDType i = 0; i < m; i++) {
      col[i] = edges[i].second;
      row_ptr[edges[i].first]++;
    }

    for (IDType i = 1; i <= n; i++) {
      row_ptr[i] += row_ptr[i - 1];
    }

    for (IDType i = n; i > 0; i--) {
      row_ptr[i] = row_ptr[i - 1];
    }

    row_ptr[0] = 0;


    free(coo_row);
    free(coo_col);

    Sort();
    InitIds();
  }

  void CSR::Reorder(vector<IDType>& order, vector<IDType>* cut, vector<bool>* in_cut){
    // cout << "Reordering the CSR..." << endl;
    IDType* new_row_ptr = new IDType[n+1];
    new_row_ptr[0] = 0;
    IDType* new_col = new IDType[m];

    vector<IDType> new_order;
    if(RERANK_CUT && cut != nullptr){
      new_order.reserve(n);
      
      for(IDType u : order){
        if(! ((*in_cut)[u])){
          new_order.push_back(u);
        }
      }

      // cout << "Removed: " << order.size() - new_order.size() << endl;
      // cout << "Cut: " << cut->size() << endl;

      new_order.insert(new_order.end(), cut->begin(), cut->end());
    } else {
      new_order = order;
    }


    vector<IDType> ranks(n);

    #pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
    for(int i=0; i<new_order.size(); i++){
      ranks[new_order[i]] = i;
    }
    
    // unordered_set<IDType> ranks_set;
    // ranks_set.insert(ranks.begin(), ranks.end());
    // cout << "Ranks: " << ranks.size() << endl;
    // cout << "Ranks Set: " << ranks_set.size() << endl;

    size_t last_index = 0;
    for(IDType i=0; i<n; i++){
      IDType u = new_order[i];
      IDType start = row_ptr[u];
      IDType end = row_ptr[u+1];
      IDType size = end-start;

      copy(col + start, col + end, new_col + last_index);
      last_index += size;
      new_row_ptr[i+1] = last_index;
    }


  #pragma omp parallel for default(shared) schedule(SCHEDULE) num_threads(NUM_THREADS)
  for(IDType i=0; i<n; i++){
    ids[ranks[i]] = i;
    inv_ids[i] = ranks[i]; 
  }

  #pragma omp parallel for default(shared) schedule(SCHEDULE) num_threads(NUM_THREADS)
    for(IDType i=0; i<m; i++){
      new_col[i] = ranks[new_col[i]];
    }

    delete[] row_ptr;
    delete[] col;

    row_ptr = new_row_ptr;
    col = new_col;

    Sort();

  }


void CSR::Sort(){
  #pragma omp parallel for default(shared) schedule(SCHEDULE) num_threads(NUM_THREADS)
    for(IDType u = 0; u < n; u++){
      IDType start = row_ptr[u];
      IDType end = row_ptr[u+1];
      sort(col + start, col + end);
    }
}


void CSR::InitIds(){
  ids = new IDType[n];
  inv_ids = new IDType[n];
  type = new char[n];

#pragma omp parallel for default(shared) schedule(SCHEDULE) num_threads(NUM_THREADS)
  for(IDType i=0; i<n; i++){
    ids[i] = i;
    inv_ids[i] = i; 
    type[i] = 0;
  }
}


void CSR::ComputeF1F2(vector<size_t>& f1, vector<size_t>& f2){
	long long s = 0;
  long long *nows = new long long[m+n+1];
  int *nowt = new int[m+n+1];
  memset( nowt, 0, sizeof(int) * (m+n+1) );

  for( IDType v = 0; v < n; ++v ){
    IDType start = row_ptr[v];
    IDType end = row_ptr[v+1];
    for( IDType i = start; i < end; ++i ) {
      int u = col[i];
      if( nowt[f1[u]] != (v+1) ) {
        ++s;
        nows[f1[u]] = s;
        nowt[f1[u]] = (v+1);
        f1[u] = s;
      } else f1[u] = nows[f1[u]];
    }
  }

  for( int v = 0; v < n; ++v )
    if( nowt[f1[v]] != -1 ) {
      nows[f1[v]] = v;
      nowt[f1[v]] = -1;
      f1[v] = v;
    } else f1[v] = nows[f1[v]];

  s = 0;
  memset( nowt, 0, sizeof(int) * (m+n+1) );
  for( int v = 0; v < n; ++v ){
    IDType start = row_ptr[v];
    IDType end = row_ptr[v+1];
    for( int i = start; i <= end; ++i ) {
      int u = (i == end) ? v : col[i];
      if( nowt[f2[u]] != (v+1) ) {
        ++s;
        nows[f2[u]] = s;
        nowt[f2[u]] = (v+1);
        f2[u] = s;
      } else f2[u] = nows[f2[u]];
    }
  }

  for( int v = 0; v < n; ++v )
    if( nowt[f2[v]] != -1 ) {
      nows[f2[v]] = v;
      nowt[f2[v]] = -1;
      f2[v] = v;
    } else f2[v] = nows[f2[v]];

  delete[] nows; delete[] nowt;
}


void CSR::Compress(vector<bool>& in_cut){
 
  vector<size_t> f1(n, 0);
  vector<size_t> f2(n, 0);

  ComputeF1F2(f1, f2);    

  vector<IDType> new_row_ptr;
  new_row_ptr.reserve(n+1);
  new_row_ptr.push_back(0);
  vector<IDType> new_col;

  IDType curr_row_ptr = 0;
  for(IDType u = 0; u < n; u++){

    if(in_cut[u] || (f1[u] == u && f2[u] == u) || row_ptr[u] == row_ptr[u+1]){

      IDType start = row_ptr[u];
      IDType end = row_ptr[u+1];

      size_t inserted = 0;
      for(IDType i = start; i < end; i++){
        IDType v = col[i];

        if(in_cut[v] || (f1[v] == v && f2[v] == v)){
          new_col.push_back(v);
          inserted++;
        }
      }

      curr_row_ptr += inserted;
      new_row_ptr.push_back(curr_row_ptr);

    } else {
      new_row_ptr.push_back(curr_row_ptr);
      IDType curr_id = u;
      IDType real_id = ids[u];

      IDType new_curr_id = (f1[u] != u) ? f1[u] : f2[u];
      IDType new_real_id = ids[new_curr_id];

      ids[curr_id] = new_real_id;
      inv_ids[real_id] = new_curr_id;
      type[real_id] = (f1[u] != u) ? 2 : 1;
    }

  }

  delete[] row_ptr;
  delete[] col;

  row_ptr = new IDType[new_row_ptr.size()];
  col = new IDType[new_col.size()];
  copy(new_row_ptr.begin(), new_row_ptr.end(), row_ptr);
  copy(new_col.begin(), new_col.end(), col);

  cout << "Compression removed " << m - new_col.size() << " edges" << endl;

  n = new_row_ptr.size() - 1;
  m = new_col.size();

  
}
