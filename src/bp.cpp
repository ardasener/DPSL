#include "bp.h"
#include <queue>
#include "external/order/order.hpp"


BP::BP(vector<uint64_t>& bp_0_sets, vector<uint64_t>& bp_1_sets, vector<uint8_t>& bp_dists, vector<bool>& used)
: bp_0_sets(bp_0_sets), bp_1_sets(bp_1_sets), bp_dists(bp_dists), used(used) {}

bool BP::PruneByBp(IDType u, IDType v, int d){
  
  int u_index = u*N_ROOTS;
  int v_index = v*N_ROOTS;
  for (int i = 0; i < N_ROOTS; ++i) {

      int td = bp_dists[u_index] + bp_dists[v_index];

      if(td <= d){
        return true;
      }

      if (td - 2 <= d)
        
        if(bp_0_sets[u_index] & bp_0_sets[v_index]){
          td -= 2;
        } else if((bp_0_sets[u_index] & bp_1_sets[v_index]) | bp_1_sets[u_index] & bp_0_sets[v_index]){
          td -= 1;
        }
  
      if (td <= d){
	      /* printf("BPPruned u=%d v=%d td=%d d=%d \n", u,v,td,d); */
	      return true;
      }

      u_index++;
      v_index++;
    }

    return false;

}

int BP::QueryByBp(IDType u, IDType v){

  int d = MAX_DIST;

  int u_index = u*N_ROOTS;
  int v_index = v*N_ROOTS;
  for (int i = 0; i < N_ROOTS; ++i) {

      int td = bp_dists[u_index] + bp_dists[v_index];

      if (td - 2 <= d)
        
        if(bp_0_sets[u_index] & bp_0_sets[v_index]){
          td -= 2;
        } else if((bp_0_sets[u_index] & bp_1_sets[v_index]) | bp_1_sets[u_index] & bp_0_sets[v_index]){
          td -= 1;
        }
  
      if (td <= d){
        d = td;
      }

      u_index++;
      v_index++;
    }

    return d;
}

void BP::InitBPForRoot(IDType r, vector<IDType>& Sr, int bp_index, CSR& csr){

  vector<pair<uint64_t,uint64_t>> new_bp_sets(csr.n, make_pair((uint64_t) 0, (uint64_t) 0));
  vector<uint8_t> new_bp_dists(csr.n, (uint8_t) MAX_DIST);

  IDType* Q0 = new IDType[csr.n];
  IDType* Q1 = new IDType[csr.n];
  IDType Q0_size = 0;
  IDType Q0_ptr = 0;
  IDType Q1_size = 0;
  IDType Q1_ptr = 0;

  Q0[Q0_size++] = r;
  new_bp_dists[r] = (uint8_t) 0;

  for(IDType i=0; i<Sr.size(); i++){
    IDType v = Sr[i];
    Q1[Q1_size++] = v;
    new_bp_dists[v] = (uint8_t) 1;
    new_bp_sets[v].first |= 1ULL << i;
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

        if(new_bp_dists[u] == (uint8_t) MAX_DIST){
          E1.emplace_back(v,u);
          new_bp_dists[u] = (uint8_t) (new_bp_dists[v] + 1);
          Q1[Q1_size++] = u;
        } else if(new_bp_dists[u] == new_bp_dists[v] + 1){
          E1.emplace_back(v,u);
        } else if (new_bp_dists[u] == new_bp_dists[v]){
          E0.emplace_back(v,u);
        } 
      } 
    }

    for(auto& p : E0){
      IDType v = p.first;
      IDType u = p.second;
      new_bp_sets[v].second |= new_bp_sets[u].first;
      new_bp_sets[u].second |= new_bp_sets[v].first;
    }

    for(auto& p : E1){
      IDType v = p.first;
      IDType u = p.second;
      new_bp_sets[u].first |= new_bp_sets[v].first;
      new_bp_sets[u].second |= new_bp_sets[v].second & ~new_bp_sets[v].first;
      //bp_sets[u].second |= bp_sets[v].second;
    }

    swap(Q0, Q1);
    swap(Q0_ptr, Q1_ptr);
    swap(Q0_size, Q1_size);
    Q1_ptr = 0;
    Q1_size = 0;
  }


  for(IDType i=0; i<csr.n; i++){

    bp_0_sets[i*N_ROOTS + bp_index] = new_bp_sets[i].first;
    bp_1_sets[i*N_ROOTS + bp_index] = new_bp_sets[i].second;
    bp_dists[i*N_ROOTS + bp_index] = new_bp_dists[i];

    // auto& bp = bp_labels[i];
    // bp.bp_dists[bp_index] = bp_dists[i];
    // bp.bp_sets[bp_index][0] = bp_sets[i].first;
    // bp.bp_sets[bp_index][1] = bp_sets[i].second;
  }

  delete[] Q0;
  delete[] Q1;

}

BP::BP(CSR& csr, vector<IDType>* cut, Mode mode) {

  double start_time = omp_get_wtime();

  bp_0_sets.resize(csr.n*N_ROOTS, 0);
  bp_1_sets.resize(csr.n*N_ROOTS, 0);
  bp_dists.resize(csr.n*N_ROOTS, MAX_DIST);

  vector<int> bp_order;
  vector<int> bp_ranks;

  if constexpr(BP_ORDER_METHOD != "same"){
    bp_order = gen_order<IDType>(csr.row_ptr, csr.col, csr.n, csr.m, BP_ORDER_METHOD);
    bp_ranks.resize(csr.n);

    #pragma omp parallel for default(shared) num_threads(NUM_THREADS) 
    for(IDType i=0; i<csr.n; i++){
          bp_ranks[bp_order[i]] = i;
    } 
  
  }

  used.resize(csr.n, false);
  vector<IDType> roots;
  roots.reserve(N_ROOTS);

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
  for(IDType i=0; i<csr.n; i++){
    IDType start = csr.row_ptr[i];
    IDType end = csr.row_ptr[i+1];

    if(end-start > 64)
      sort(csr.col+start, csr.col+end, [&bp_ranks](IDType u, IDType v){
        if constexpr(BP_ORDER_METHOD == "same"){
	        return u > v;
        } else {
          return bp_ranks[u] > bp_ranks[v];
        }
	    });
  }

  priority_queue<pair<int64_t, IDType>, vector<pair<int64_t, IDType>>, less<pair<int64_t, IDType>>> candidates;
  if constexpr(BP_RERANK){
    vector<int64_t> ngh_ranks(csr.n, 0);

    #pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(SCHEDULE)
    for(IDType i=0; i<csr.n; i++){
      
      IDType start = csr.row_ptr[i];
      IDType end = csr.row_ptr[i+1];

      if constexpr(BP_ORDER_METHOD == "same"){
        ngh_ranks[i] = i;
      } else{
        ngh_ranks[i] += bp_ranks[i];
      }

      for(IDType j=start; j < end; j++){
        IDType v = csr.col[j];
        
        if constexpr(BP_ORDER_METHOD == "same"){
          ngh_ranks[i] += v;
        } else{
          ngh_ranks[i] += bp_ranks[v];
        }
      }
    }

    if(cut == nullptr)
      for(IDType i=0; i<csr.n; i++){
        candidates.emplace(ngh_ranks[i], i);
      }
    else
      for(IDType i : *cut){
        candidates.emplace(ngh_ranks[i], i);
      }


  } else {

    if(cut == nullptr)
      for(IDType i=0; i<csr.n; i++){

        int64_t weight = 0;
        if constexpr(BP_ORDER_METHOD != "same"){
          weight = bp_ranks[i];
        }

        candidates.emplace(weight, i);
      }
    else
      for(IDType i : *cut){
        
        int64_t weight = 0;
        if constexpr(BP_ORDER_METHOD != "same"){
          weight = bp_ranks[i];
        }

        candidates.emplace(0, i);
      }

  }

  vector<vector<IDType>> Srs(N_ROOTS);

  // TODO: use avg. degree of nghood for candidate selection
  // TODO: can we add the same node twice? If it has > 64 nghboors
  IDType number_of_roots_used = 0;
  for(IDType i=0; i<N_ROOTS; i++){

    int64_t bp_rank = -1;
    IDType root = -1;
    while(! candidates.empty()){
      auto& p = candidates.top();
      bp_rank = p.first;
      root = p.second;
      candidates.pop();

      if(!used[root]){
        number_of_roots_used++;
        break;
      }

    }

    if(candidates.empty()){
      cout << "Run out of root nodes for BP" << endl;
      break;
    }

    roots.push_back(root);
    used[root] = true;
    bp_rank -= root;
    
    Srs[i].reserve(64);

    IDType start = csr.row_ptr[root];
    IDType end = csr.row_ptr[root+1];

    for(IDType j=start; j<end; j++){
      IDType v = csr.col[j];
      if(!used[v]){
        bp_rank -= v;
        Srs[i].push_back(v);
        used[v] = true;
        if(Srs[i].size() == 64){
          break;
        }
      }
    }

    if constexpr(ALLOW_DUPLICATE_BP){
      candidates.emplace(bp_rank, root);
    }
  }

  cout << "BP Root Count: " << roots.size() << endl;

#pragma omp parallel for default(shared) num_threads(NUM_THREADS) schedule(dynamic)
  for(int i=0; i<number_of_roots_used; i++){
    vector<IDType>& Sr = Srs[i];
    IDType root = roots[i];
    InitBPForRoot(root, Sr, i, csr);    
  }


  double end_time = omp_get_wtime();
  cout << "Constructed BP Labels in " << end_time-start_time << " seconds" << endl;

/*
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
*/

}