#include "cut.h"

#include <omp.h>
#include <stdio.h>

#include "mtmetis.h"

VertexCut::~VertexCut() {
  if (partition != nullptr) delete[] partition;
}

VertexCut *VertexCut::Partition(CSR &csr, string partitioner, string params,
                                string order_method, int np) {
  VertexCut *vc = new VertexCut;

  if (partitioner == "mtmetis") {
    const uint32_t nvtxs = csr.n;
    const uint32_t ncon = csr.m;
    uint32_t *xadj = (uint32_t *)csr.row_ptr;
    uint32_t *adj = (uint32_t *)csr.col;
    int32_t *vwgt = new int32_t[csr.n];
    const uint32_t vsize = csr.n;  // Unused
    int32_t *adjwgt = nullptr;
    const uint32_t nparts = np;
    float *tpwgts = nullptr;
    const float ubvec = 1;
    int32_t r_edgecut;
    uint32_t *where = new uint32_t[csr.n];

#pragma omp parallel for num_threads(NUM_THREADS)
    for (IDType i = 0; i < csr.n; i++) {
      vwgt[i] = csr.row_ptr[i + 1] - csr.row_ptr[i];
    }

    double *options = mtmetis_init_options();
    options[MTMETIS_OPTION_NTHREADS] = (double)NUM_THREADS;
    options[MTMETIS_OPTION_SEED] = (double)42;

    double part_time = omp_get_wtime();
    MTMETIS_PartGraphKway(&nvtxs, &ncon, xadj, adj, vwgt, &vsize, adjwgt,
                          &nparts, tpwgts, &ubvec, options, &r_edgecut, where);
    cout << "Partition Time: " << omp_get_wtime() - part_time << " seconds"
         << endl;

    delete[] vwgt;

    vc->partition = (IDType *)where;

    cout << "Ordering..." << endl;
    vc->order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

    cout << "Ranking..." << endl;
    vc->ranks.resize(csr.n);
    for (IDType i = 0; i < csr.n; i++) {
      vc->ranks[vc->order[i]] = i;
    }

    vc->Init(csr, np);

  } else {
    cerr << "Currently only mtmetis partitioner is supported" << endl;
    throw -1;
  }

  return vc;
}

void VertexCut::Init(CSR &csr, int np) {
  vector<bool> in_cut(csr.n, false);
  cout << "Calculating cut..." << endl;
  for (IDType i = csr.n - 1; i >= 0; i--) {
    IDType u = order[i];
    IDType start = csr.row_ptr[u];
    IDType end = csr.row_ptr[u + 1];

    for (IDType j = start; j < end; j++) {
      IDType v = csr.col[j];

      if (partition[u] != partition[v]) {
        if (in_cut[v] || in_cut[u]) {
          continue;
        }

        if (ranks[u] > ranks[v]) {
          cut.insert(u);
          in_cut[u] = true;
        } else {
          cut.insert(v);
          in_cut[v] = true;
        }
      }
    }
  }

  cout << "Cut Size: " << cut.size() << endl;

  cout << "Calculating edges and nodes..." << endl;
  vector<vector<pair<IDType, IDType>>> edges(np);
  for (IDType u = 0; u < csr.n; u++) {
    IDType start = csr.row_ptr[u];
    IDType end = csr.row_ptr[u + 1];

    bool u_in_cut = in_cut[u];

    for (int j = start; j < end; j++) {
      IDType v = csr.col[j];

      bool v_in_cut = in_cut[v];

      if (u_in_cut && v_in_cut) {
        for (int i = 0; i < np; i++) {
          edges[i].emplace_back(u, v);
        }
      } else if (partition[u] == partition[v] || v_in_cut) {
        edges[partition[u]].emplace_back(u, v);
      } else if (u_in_cut) {
        edges[partition[v]].emplace_back(u, v);
      }
    }
  }

  for (int i = 0; i < np; i++) {
    sort(edges[i].begin(), edges[i].end(), less<pair<IDType, IDType>>());
    auto unique_it = unique(
        edges[i].begin(), edges[i].end(),
        [](const pair<IDType, IDType> &p1, const pair<IDType, IDType> &p2) {
          return (p1.first == p2.first) && (p1.second == p2.second);
        });
    edges[i].erase(unique_it, edges[i].end());
  }

  cout << "Constructing csrs..." << endl;
  csrs.resize(np, nullptr);
  for (int i = 0; i < np; i++) {
    IDType n = csr.n;
    IDType m = edges[i].size();
    IDType *row_ptr = new IDType[n + 1];
    IDType *col = new IDType[m];

    fill(row_ptr, row_ptr + n + 1, 0);

    row_ptr[0] = 0;

    IDType edge_index = 0;
    for (IDType j = 0; j < n; j++) {
      IDType count = 0;
      while (edge_index < m && edges[i][edge_index].first == j) {
        auto &edge = edges[i][edge_index];
        col[edge_index] = edge.second;
        count++;
        edge_index++;
      }
      row_ptr[j + 1] = count + row_ptr[j];
    }

    IDType *ids = new IDType[n];
    IDType *inv_ids = new IDType[n];
    IDType *comp_ids = new IDType[n];
    char *type = new char[n];

#pragma omp parallel for num_threads(NUM_THREADS) default(shared)
    for (IDType j = 0; j < n; j++) {
      ids[j] = csr.ids[j];
      inv_ids[j] = csr.inv_ids[j];
      comp_ids[j] = csr.comp_ids[j];
      type[j] = csr.type[j];
    }

    csrs[i] = new CSR(row_ptr, col, ids, inv_ids, type, comp_ids, n, m);
    cout << "M for P" << i << ": " << m << endl;
  }
}

VertexCut *VertexCut::Read(CSR &csr, string part_file, string order_method,
                           int np) {
  VertexCut *vc = new VertexCut;

  cout << "Ordering..." << endl;
  vc->order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  cout << "Ranking..." << endl;
  vc->ranks.resize(csr.n);
  for (IDType i = 0; i < csr.n; i++) {
    vc->ranks[vc->order[i]] = i;
  }

  ifstream part_ifs(part_file);

  vc->partition = new IDType[csr.n];
  auto &partition = vc->partition;
  int x;
  IDType i = 0;
  while (part_ifs >> x) {
    partition[i++] = x;
  }

  part_ifs.close();

  vc->Init(csr, np);

  return vc;
}
