#include "bp.h"


BP::BP(vector<BPLabel>& bp_labels, vector<bool>& used) : bp_labels(bp_labels), used(used){}

bool BP::PruneByBp(IDType u, IDType v, int d){

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

int BP::QueryByBp(IDType u, IDType v){

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

void BP::InitBPForRoot(IDType r, vector<IDType>& Sr, int bp_index, CSR& csr){

  vector<pair<uint64_t,uint64_t>> bp_sets(csr.n, make_pair((uint64_t) 0, (uint64_t) 0));
  vector<uint8_t> bp_dists(csr.n, (uint8_t) MAX_DIST);

  IDType* Q0 = new IDType[csr.n];
  IDType* Q1 = new IDType[csr.n];
  IDType Q0_size = 0;
  IDType Q0_ptr = 0;
  IDType Q1_size = 0;
  IDType Q1_ptr = 0;

  Q0[Q0_size++] = r;
  bp_dists[r] = (uint8_t) 0;

  for(IDType i=0; i<Sr.size(); i++){
    IDType v = Sr[i];
    Q1[Q1_size++] = v;
    bp_dists[v] = (uint8_t) 1;
    bp_sets[v].first |= 1ULL << i;
  }

  while(Q0_ptr < Q0_size){
    vector<pair<IDType,IDType>> E0;
    vector<pair<IDType,IDType>> E1;

    while(Q0_ptr < Q0_size){
      IDType v = Q0[Q0_ptr++];

      IDType start = csr.row_ptr[v];
      IDType end = csr.row_ptr[v+1];

      for(IDType i=start; i<end; i++){
	IDType u = csr.col[i];

	if(bp_dists[u] == (uint8_t) MAX_DIST){
	  E1.emplace_back(v,u);
	  bp_dists[u] = (uint8_t) (bp_dists[v] + 1);
	  Q1[Q1_size++] = u;
	} else if(bp_dists[u] == bp_dists[v] + 1){
	  E1.emplace_back(v,u);
	} else if (bp_dists[u] == bp_dists[v]){
	  E0.emplace_back(v,u);

	} 
      } 
    }

    for(auto& p : E0){
      IDType v = p.first;
      IDType u = p.second;
      bp_sets[v].second |= bp_sets[u].first;
      bp_sets[u].second |= bp_sets[v].first;
    }

    for(auto& p : E1){
      IDType v = p.first;
      IDType u = p.second;
      bp_sets[u].first |= bp_sets[v].first;
      bp_sets[u].second |= bp_sets[v].second & ~bp_sets[v].first;
      //bp_sets[u].second |= bp_sets[v].second;
    }

    swap(Q0, Q1);
    swap(Q0_ptr, Q1_ptr);
    swap(Q0_size, Q1_size);
    Q1_ptr = 0;
    Q1_size = 0;
  }


  for(IDType i=0; i<csr.n; i++){
    auto& bp = bp_labels[i];
    bp.bp_dists[bp_index] = bp_dists[i];
    bp.bp_sets[bp_index][0] = bp_sets[i].first;
    bp.bp_sets[bp_index][1] = bp_sets[i].second;
  }

  delete[] Q0;
  delete[] Q1;

}

BP::BP(CSR& csr, vector<IDType>& ranks, vector<IDType>& order, vector<IDType>* cut, Mode mode) {

  double start_time = omp_get_wtime();

  bp_labels.resize(csr.n);
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
  for(IDType i=0; i<csr.n; i++){
    for(int j=0; j<N_ROOTS; j++){
      bp_labels[i].bp_dists[j] = (uint8_t) MAX_DIST;
      bp_labels[i].bp_sets[j][0] = (uint64_t) 0;
      bp_labels[i].bp_sets[j][1] = (uint64_t) 0;   
    }
  }


  used.resize(csr.n, false);
  vector<IDType> roots;
  roots.reserve(N_ROOTS);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(runtime)
  for(IDType i=0; i<csr.n; i++){
    IDType start = csr.row_ptr[i];
    IDType end = csr.row_ptr[i+1];

    if(end-start > 64)
      sort(csr.col+start, csr.col+end, [&ranks](IDType u, IDType v){
	  return ranks[u] > ranks[v];
	});
  }

  vector<IDType> candidates = order;



  // If the mode is global then the cut vertices are inserted to the end of the list
  // This way they are picked first and only when they are covered we start choosing others
  // If the mode is local then the cut vertices are ignored entirely here (ie, marked used)
  if(cut != nullptr){
    if(mode == GLOBAL_BP_MODE){
      candidates.insert(candidates.end(), cut->rbegin(), cut->rend());
    } else {
      for(IDType cut_node : *cut){
	      used[cut_node] = true;
      }
    }

  } 

  vector<vector<IDType>> Srs(N_ROOTS);

  IDType root_index = candidates.size()-1;
  IDType number_of_roots_used = 0;
  for(IDType i=0; i<N_ROOTS; i++){


    while(root_index >= 0){
      IDType root = candidates[root_index]; 
      if(!used[root]){
        number_of_roots_used++;
        break;
      }
      root_index--;
    }

    if(root_index == 0){
      cout << "Run out of root nodes for BP" << endl;
      break;
    }

    IDType root = candidates[root_index];
    roots.push_back(root);
    /* cout << "Chosen root=" << root << endl; */
    /* cout << "Chosen root rank=" << ranks[root] << endl; */
    used[root] = true;
    
    Srs[i].reserve(64);

    IDType start = csr.row_ptr[root];
    IDType end = csr.row_ptr[root+1];

    for(IDType j=start; j<end && j-start < 64; j++){
      IDType v = csr.col[j];
      if(!used[v]){
        Srs[i].push_back(v);
        used[v] = true;
      }
    }
  }

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(dynamic)
  for(int i=0; i<number_of_roots_used; i++){
    vector<IDType>& Sr = Srs[i];
    IDType root = roots[i];
    InitBPForRoot(root, Sr, i, csr);    
  }


  double end_time = omp_get_wtime();
  cout << "Constructed BP Labels in " << end_time-start_time << " seconds" << endl;

#ifdef DEBUG
  ofstream ofs("output_bp_labels.txt");
  ofs << "__Roots__" << endl;
  for(int i=0; i<roots.size(); i++){
    ofs << roots[i] << ", ";	
  }
  ofs << endl;
  ofs << "__Dists__" << endl;
  for(IDType v=0; v<csr.n; v++){
	  ofs << v << ": ";
	  for(int i=0; i<N_ROOTS; i++){
		  ofs << (int) bp_labels[v].bp_dists[i] << ", ";	
	  }
	  ofs << endl;
  }
  
  ofs << "__Sets__" << endl;
  for(IDType v=0; v<csr.n; v++){
	  ofs << v << ": ";
	  for(int i=0; i<N_ROOTS; i++){
		  ofs << "(" << bp_labels[v].bp_sets[i][0] << "," << bp_labels[v].bp_sets[i][1] << ")" << ", ";	
	  }
	  ofs << endl;
  }

  ofs.close();


#endif

}

