#include "cut.h"

VertexCut::~VertexCut() {
  if (partition != nullptr)
    delete[] partition;
}

VertexCut *VertexCut::Partition(CSR &csr, string partitioner, string params,
                                string order_method, int np) {
  ofstream ofs("temp.graph");
  ofs << csr.n << " " << csr.m / 2 << " " << 10 << endl;

  for (IDType i = 0; i < csr.n; i++) {

    IDType start = csr.row_ptr[i];
    IDType end = csr.row_ptr[i + 1];

    ofs << end - start << " ";
    // ofs << 1 << " ";

    for (IDType j = start; j < end; j++) {
      IDType v = csr.col[j];
      ofs << v + 1 << " ";
    }

    ofs << endl;
  }

  ofs.close();

  stringstream ss;
  int ret_val;

  if (partitioner.find("gpmetis") != string::npos) {
    ss << partitioner << " --objtype=cut temp.graph " << np;
    ss << " " << params;
    ss << " && "
       << "mv temp.graph.part." << np << " temp.part";
    cout << "Running: " << ss.str() << endl;
    ret_val = system(ss.str().c_str());
  } else if (partitioner.find("mtmetis") != string::npos) {
    ss << partitioner << " temp.graph " << np << " temp.part";
    ss << " " << params;
    cout << "Running: " << ss.str() << endl;
    ret_val = system(ss.str().c_str());
  } else if (partitioner.find("pulp") != string::npos) {
    ss << partitioner << " temp.graph " << np << " -o temp.part";
    ss << " " << params;
    cout << "Running: " << ss.str() << endl;
    ret_val = system(ss.str().c_str());
  } else {
    throw "Unknown partitioner";
  }

  cout << "Return Value: " << ret_val << endl;
  VertexCut *vc = VertexCut::Read(csr, "temp.part", order_method, np);

  ret_val = system("rm temp.part temp.graph");

  return vc;
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

  cout << "Final i=" << i << endl;

  vector<bool> in_cut(csr.n, false);
  cout << "Calculating cut..." << endl;
  for (IDType i = csr.n - 1; i >= 0; i--) {
    IDType u = vc->order[i];
    IDType start = csr.row_ptr[u];
    IDType end = csr.row_ptr[u + 1];

    for (IDType j = start; j < end; j++) {
      IDType v = csr.col[j];

      if (partition[u] != partition[v]) {

        if (in_cut[v] || in_cut[u]) {
          continue;
        }

        if (vc->ranks[u] > vc->ranks[v]) {
          vc->cut.insert(u);
          in_cut[u] = true;
        } else {
          vc->cut.insert(v);
          in_cut[v] = true;
        }
      }
    }
  }

  cout << "Cut Size: " << vc->cut.size() << endl;

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
  vc->csrs.resize(np, nullptr);
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

    vc->csrs[i] = new CSR(row_ptr, col, ids, inv_ids, type, comp_ids, n, m);
    cout << "M for P" << i << ": " << m << endl;
  }

  return vc;
}
