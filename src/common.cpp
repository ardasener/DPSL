#include "common.h"

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
}

  CSR::CSR(CSR& csr){
    row_ptr = new IDType[csr.n+1];
    col = new IDType[csr.m];

    for(IDType i=0; i<csr.n+1; i++){
      row_ptr[i] = csr.row_ptr[i];
    }

    for(IDType i=0; i<csr.m; i++){
      col[i] = csr.col[i];
    }

    n = csr.n;
    m = csr.m;

  }

  CSR::CSR(IDType * row_ptr, IDType *col, IDType n, IDType m): 
    row_ptr(row_ptr), col(col), n(n), m(m) {}

  CSR::CSR(string filename) {

    bool is_mtx = true;

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


    delete[] coo_row;
    delete[] coo_col;
  }

  void CSR::Reorder(vector<IDType>& order, vector<IDType>* cut, vector<bool>* in_cut){
    cout << "Reordering the CSR..." << endl;
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

      cout << "Removed: " << order.size() - new_order.size() << endl;
      cout << "Cut: " << cut->size() << endl;

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

  real_ids = new IDType[n];
  reorder_ids = new IDType[n];

  #pragma omp parallel for default(shared) schedule(SCHEDULE) num_threads(NUM_THREADS)
  for(IDType i=0; i<n; i++){
    real_ids[ranks[i]] = i;
    reorder_ids[i] = ranks[i]; 
  }

  #pragma omp parallel for default(shared) schedule(SCHEDULE) num_threads(NUM_THREADS)
    for(IDType i=0; i<m; i++){
      new_col[i] = ranks[new_col[i]];
    }

    delete[] row_ptr;
    delete[] col;

    row_ptr = new_row_ptr;
    col = new_col;

  }