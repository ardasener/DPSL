#ifndef BP_H
#define BP_H

#include "common.h"

using namespace std;

struct BPLabel{
  uint8_t bp_dists[N_ROOTS];
  uint64_t bp_sets[N_ROOTS][2];
};

class BP {
public:
    vector<BPLabel> bp_labels;
    CSR& csr;
  
    BP(CSR& csr, vector<int>& ranks, vector<int>& order);
    void InitBPForRoot(int r, vector<int>& Sr, int root_index);
    bool PruneByBp(int u, int v, int d);
    int QueryByBp(int u, int v);
};

inline bool BP::PruneByBp(int u, int v, int d){

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

inline int BP::QueryByBp(int u, int v){

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

inline void BP::InitBPForRoot(int r, vector<int>& Sr, int bp_index){

  cout << "BP Init for root=" << r << endl;
  vector<pair<uint64_t,uint64_t>> bp_sets(csr.n, make_pair((uint64_t) 0, (uint64_t) 0));
  vector<uint8_t> bp_dists(csr.n, (uint8_t) MAX_DIST);

  int* Q0 = new int[csr.n];
  int* Q1 = new int[csr.n];
  int Q0_size = 0;
  int Q0_ptr = 0;
  int Q1_size = 0;
  int Q1_ptr = 0;

  Q0[Q0_size++] = r;
  bp_dists[r] = (uint8_t) 0;

  for(int i=0; i<Sr.size(); i++){
    int v = Sr[i];
    Q1[Q1_size++] = v;
    bp_dists[v] = (uint8_t) 1;
    bp_sets[v].first |= 1ULL << i;
  }

  while(Q0_ptr < Q0_size){
    unordered_set<pair<int,int>, pair_hash> E0;
    unordered_set<pair<int,int>, pair_hash> E1;

    while(Q0_ptr < Q0_size){
      int v = Q0[Q0_ptr++];

      int start = csr.row_ptr[v];
      int end = csr.row_ptr[v+1];

      for(int i=start; i<end; i++){
	int u = csr.col[i];

	if(bp_dists[u] == (uint8_t) MAX_DIST){
	  E1.emplace(v,u);
	  bp_dists[u] = (uint8_t) (bp_dists[v] + 1);
	  Q1[Q1_size++] = u;
	} else if(bp_dists[u] == bp_dists[v] + 1){
	  E1.emplace(v,u);
	} else if (bp_dists[u] == bp_dists[v]){
	  E0.emplace(v,u);
	} 
      } 
    }

    for(auto& p : E0){
      int v = p.first;
      int u = p.second;
      bp_sets[v].second |= bp_sets[u].first;
      bp_sets[u].second |= bp_sets[v].first;
    }

    for(auto& p : E1){
      int v = p.first;
      int u = p.second;
      bp_sets[u].first |= bp_sets[v].first;
      bp_sets[u].second |= bp_sets[v].second & ~bp_sets[v].first;
      /* bp_sets[u].second |= bp_sets[v].second; */
    }

    swap(Q0, Q1);
    swap(Q0_ptr, Q1_ptr);
    swap(Q0_size, Q1_size);
    Q1_ptr = 0;
    Q1_size = 0;
  }


  for(int i=0; i<csr.n; i++){
    auto& bp = bp_labels[i];
    bp.bp_dists[bp_index] = bp_dists[i];
    bp.bp_sets[bp_index][0] = bp_sets[i].first;
    bp.bp_sets[bp_index][1] = bp_sets[i].second;
  }

  delete[] Q0;
  delete[] Q1;

}

BP::BP(CSR& csr, vector<int>& ranks, vector<int>& order): csr(csr){
 
  cout << "Creating BP Labels" << endl; 
  bp_labels.resize(csr.n);
  for(int i=0; i<csr.n; i++){
    for(int j=0; j<N_ROOTS; j++){
      bp_labels[i].bp_dists[j] = (uint8_t) MAX_DIST;
      bp_labels[i].bp_sets[j][0] = (uint64_t) 0;
      bp_labels[i].bp_sets[j][1] = (uint64_t) 0;   
    }
  }


  vector<bool> used(csr.n, false);
  vector<int> roots;

  cout << "Sorting the neighbors" << endl;
  for(int i=0; i<csr.n; i++){
    int start = csr.row_ptr[i];
    int end = csr.row_ptr[i+1];

    sort(csr.col+start, csr.col+end, [ranks](int u, int v){
	return ranks[u] > ranks[v];
      });
  }


  int root_index = csr.n-1;
  for(int i=0; i<N_ROOTS; i++){

    cout << "Choosing root " << i << endl;
    vector<int> SR;
    SR.reserve(64);

    while(root_index >= 0){
      int root = order[root_index]; 
      if(!used[root]){
	break;
      }
      root_index--;
    }

    if(root_index == 0){
      cout << "Run out of root nodes for BP" << endl;
      break;
    }

    int root = order[root_index];
    roots.push_back(root);
    cout << "Chosen root=" << root << endl;
    cout << "Chosen root rank=" << ranks[root] << endl;
    used[root] = true;

    int start = csr.row_ptr[root];
    int end = csr.row_ptr[root+1];

    cout << "Computing SR" << endl;
    for(int j=start; j<end && j-start < 64; j++){
      int v = csr.col[j];
      SR.push_back(v);
      used[v] = true;
    }

    InitBPForRoot(root, SR, i);    
  }

#ifdef DEBUG
  ofstream ofs("output_bp_labels.txt");
  ofs << "__Roots__" << endl;
  for(int i=0; i<roots.size(); i++){
    ofs << roots[i] << ", ";	
  }
  ofs << endl;
  ofs << "__Dists__" << endl;
  for(int v=0; v<csr.n; v++){
	  ofs << v << ": ";
	  for(int i=0; i<N_ROOTS; i++){
		  ofs << (int) bp_labels[v].bp_dists[i] << ", ";	
	  }
	  ofs << endl;
  }
  
  ofs << "__Sets__" << endl;
  for(int v=0; v<csr.n; v++){
	  ofs << v << ": ";
	  for(int i=0; i<N_ROOTS; i++){
		  ofs << "(" << bp_labels[v].bp_sets[i][0] << "," << bp_labels[v].bp_sets[i][1] << ")" << ", ";	
	  }
	  ofs << endl;
  }

  ofs.close();




#endif

}


#endif