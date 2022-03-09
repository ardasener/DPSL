#include "external/order/order.hpp"
#include "external/pigo/pigo.hpp"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>

#define N_ROOTS 16

using namespace std;

using IDType = int;

using PigoCOO = pigo::COO<IDType, IDType, IDType *, true, true, true, false,
                          float, float *>;

const char MAX_DIST = CHAR_MAX;

struct CSR {
  IDType *row_ptr;
  IDType *col;
  IDType n;
  IDType m;

  CSR(string filename) {
    PigoCOO pigo_coo(filename);

    IDType *coo_row = pigo_coo.x();
    IDType *coo_col = pigo_coo.y();
    m = pigo_coo.m();
    n = pigo_coo.n();

    vector<pair<IDType, IDType>> edges(m);

    for (size_t i = 0; i < m; i++) {
      edges[i] = pair<IDType, IDType>(coo_row[i], coo_col[i]);
    }

    sort(edges.begin(), edges.end(), less<pair<IDType, IDType>>());

    row_ptr = new IDType[n + 1];
    col = new IDType[m];

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
};

// Bit-Parallel Labels
struct BPLabel {
  char bpspt_d[N_ROOTS];
  uint64_t bpspt_s[N_ROOTS][2];
};

// Stores the labels for each vertex
struct LabelSet {
  vector<int> vertices;
  vector<int> dist_ptrs;
};

class PSL {

private:
  CSR &csr;
  vector<LabelSet> labels;
  vector<int> &ranks;
  vector<BPLabel> bp_labels;
  vector<bool> usd_bp;
  vector<int> v_vs[N_ROOTS];

  void ConstructBPLabel();
  int BPQuery(int u, int v);
  bool BPPrune(int u, int v, int d);
  bool Prune(int u, int v, int d, vector<char> &cache);
  bool Pull(int u, int d);

public:
  PSL(CSR &csr_, vector<int> &ranks_) : csr(csr_), ranks(ranks_), labels(csr.n) {
    ConstructBPLabel();
  }

  void Index();
  vector<int> Query(int u);
};

vector<int> PSL::Query(int u) {

  vector<int> results;

  auto &labels_u = labels[u];

  vector<char> cache(csr.n, MAX_DIST);

  for (int i = 0; i < MAX_DIST; i++) {
    int dist_start = labels_u.dist_ptrs[i];
    int dist_end = labels_u.dist_ptrs[i + 1];

    for (int j = dist_start; j < dist_end; j++) {
      int w = labels_u.vertices[j];
      cache[w] = i;
    }
  }

  for (int v = 0; v < csr.n; v++) {

    auto& labels_v = labels[v];
    int min = MAX_DIST;

    for (int i = 0; i < min; i++) {
      int dist_start = labels_v.dist_ptrs[i];
      int dist_end = labels_v.dist_ptrs[i + 1];

      for (int j = dist_start; j < dist_end; j++) {
        int w = labels_v.vertices[j];
          
        int dist = i + cache[w];
        if(dist > min){
          min = dist;
        }
      }
    }
    
    results.push_back(min);

  }

  return results;
}

bool PSL::Prune(int u, int v, int d, vector<char> &cache) {

  auto &labels_v = labels[v];

  for (int i = 0; i < d; i++) {
    int dist_start = labels_v.dist_ptrs[i];
    int dist_end = labels_v.dist_ptrs[i + 1];

    for (int j = dist_start; j < dist_end; j++) {
      int w = labels_v.vertices[j];

      if (i + cache[w] <= d) {
        return true;
      }
    }
  }

  return false;
}

bool PSL::Pull(int u, int d) {

  bool updated = true;

  int start = csr.row_ptr[u];
  int end = csr.row_ptr[u + 1];

  vector<char> cache(csr.n, MAX_DIST);

  auto &labels_u = labels[u];
  for (int i = 0; i < d; i++) {
    int dist_start = labels_u.dist_ptrs[i];
    int dist_end = labels_u.dist_ptrs[i + 1];

    for (int j = dist_start; j < dist_end; j++) {
      int w = labels_u.vertices[j];
      cache[w] = i;
    }
  }

  for (int i = start; i < end; i++) {
    int v = csr.col[i];
    auto &labels_v = labels[v];

    for (int j = labels_v.dist_ptrs[d - 1]; j < labels_v.vertices.size(); j++) {
      int w = labels_v.vertices[j];

      if (ranks[u] > ranks[w]) {
        continue;
      }

      if (BPPrune(u, w, d) || Prune(u, w, d, cache)) {
        continue;
      }

      labels_u.vertices.push_back(w);
      cache[w] = d;
      updated = true;
    }
  }

  labels_u.dist_ptrs.push_back(labels_u.vertices.size());

  return updated;
}

void PSL::Index() {

  // Adds the first two level of vertices
  // Level 0: vertex to itself
  // Level 1: vertex to neighbors
  for (int u = 0; u < csr.n; u++) {
    auto &labels_u = labels[u];
    labels_u.vertices.push_back(u);
    labels_u.dist_ptrs.push_back(0);
    labels_u.dist_ptrs.push_back(1);

    int start = csr.row_ptr[u];
    int end = csr.row_ptr[u + 1];

    for (int j = start; j < end; j++) {
      int v = csr.col[j];

      if (ranks[v] > ranks[u]) {
        labels_u.vertices.push_back(v);
      }
    }

    labels_u.dist_ptrs.push_back(labels_u.vertices.size());
  }

  bool updated = true;
  for (int d = 2; d < MAX_DIST && updated; d++) {
    updated = false;
    for (int u = 0; u < csr.n; u++) {
      if (Pull(u, d)) {
        updated = true;
      }
    }
  }
}

int PSL::BPQuery(int u, int v) {
  BPLabel &idx_u = bp_labels[u], &idx_v = bp_labels[v];
  int d = MAX_DIST;
  for (int i = 0; i < N_ROOTS; ++i) {
    int td = idx_u.bpspt_d[i] + idx_v.bpspt_d[i];
    if (td - 2 <= d)
      td += (idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2
            : ((idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) |
               (idx_u.bpspt_s[i][1] & idx_v.bpspt_s[i][0]))
                ? -1
                : 0;
    if (td < d)
      d = td;
  }
  return d;
}

bool PSL::BPPrune(int u, int v, int d) {
  BPLabel &idx_u = bp_labels[u], &idx_v = bp_labels[v];
  for (int i = 0; i < N_ROOTS; ++i) {
    int td = idx_u.bpspt_d[i] + idx_v.bpspt_d[i];
    if (td - 2 <= d)
      td += (idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][0]) ? -2
            : ((idx_u.bpspt_s[i][0] & idx_v.bpspt_s[i][1]) |
               (idx_u.bpspt_s[i][1] & idx_v.bpspt_s[i][0]))
                ? -1
                : 0;
    if (td <= d)
      return true;
  }
  return false;
}

void PSL::ConstructBPLabel() {
  bp_labels.resize(csr.n);
  usd_bp.resize(csr.n, false);

  int r = 0;
  for (int i = 0; i < N_ROOTS; i++) {
    while (r < csr.n && usd_bp[r]) {
      r++;
    }
    if (r == csr.n) {
      for (int v = 0; v < csr.n; v++) {
        bp_labels[v].bpspt_d[i] = MAX_DIST;
      }
    } else {
      usd_bp[r] = true;
      v_vs[i].push_back(r);
      int ns = 0;

      int start = csr.row_ptr[r];
      int end = csr.row_ptr[r + 1];

      for (int j = start; j < end; j++) {
        int v = csr.col[j];

        if (!usd_bp[v]) {
          usd_bp[v] = true;
          v_vs[i].push_back(v);
          if (++ns == 64) {
            break;
          }
        }
      }
    }
  }

  vector<char> tmp_d(csr.n);
  vector<pair<uint64_t, uint64_t>> tmp_s(csr.n);
  vector<int> que(csr.n);
  vector<pair<int, int>> child_es(csr.m / 2);

#pragma omp parallel
  {
    int pid = omp_get_thread_num();
    int nt = omp_get_num_threads();

    for (int i_bpspt = pid; i_bpspt < N_ROOTS; i_bpspt += nt) {

      if (v_vs[i_bpspt].size() == 0)
        continue;

      fill(tmp_d.begin(), tmp_d.end(), MAX_DIST);
      fill(tmp_s.begin(), tmp_s.end(), pair<uint64_t, uint64_t>(0, 0));

      r = v_vs[i_bpspt][0];
      int que_t0 = 0, que_t1 = 0, que_h = 0;
      que[que_h++] = r;
      tmp_d[r] = 0;
      que_t1 = que_h;

      for (size_t i = 1; i < v_vs[i_bpspt].size(); ++i) {
        int v = v_vs[i_bpspt][i];
        que[que_h++] = v;
        tmp_d[v] = 1;
        tmp_s[v].first = 1ULL << (i - 1);
      }

      for (int d = 0; que_t0 < que_h; ++d) {
        // int num_sibling_es = 0;
        int num_child_es = 0;

        for (int que_i = que_t0; que_i < que_t1; ++que_i) {
          int v = que[que_i];

          int start = csr.row_ptr[v];
          int end = csr.row_ptr[v + 1];

          for (int i = start; i < end; ++i) {
            int tv = csr.col[i];
            int td = d + 1;

            if (d == tmp_d[tv]) {
              if (v < tv) {
                // sibling_es[num_sibling_es].first  = v;
                // sibling_es[num_sibling_es].second = tv;
                //++num_sibling_es;
                tmp_s[v].second |= tmp_s[tv].first;
                tmp_s[tv].second |= tmp_s[v].first;
              }
            } else if (d < tmp_d[tv]) {
              if (tmp_d[tv] == MAX_DIST) {
                que[que_h++] = tv;
                tmp_d[tv] = td;
              }
              child_es[num_child_es].first = v;
              child_es[num_child_es].second = tv;
              ++num_child_es;
              // tmp_s[tv].first  |= tmp_s[v].first;
              // tmp_s[tv].second |= tmp_s[v].second;
            }
          }
        }

        /*for (int i = 0; i < num_sibling_es; ++i) {
                int v = sibling_es[i].first, w = sibling_es[i].second;
                tmp_s[v].second |= tmp_s[w].first;
                tmp_s[w].second |= tmp_s[v].first;
        }*/

        for (int i = 0; i < num_child_es; ++i) {
          int v = child_es[i].first, c = child_es[i].second;
          tmp_s[c].first |= tmp_s[v].first;
          tmp_s[c].second |= tmp_s[v].second;
        }

        que_t0 = que_t1;
        que_t1 = que_h;
      }

      for (int v = 0; v < csr.n; ++v) {
        bp_labels[v].bpspt_d[i_bpspt] = tmp_d[v];
        bp_labels[v].bpspt_s[i_bpspt][0] = tmp_s[v].first;
        bp_labels[v].bpspt_s[i_bpspt][1] = tmp_s[v].second & ~tmp_s[v].first;
      }
    }
  }
}