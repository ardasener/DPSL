#include "cut.h"

#include <omp.h>
#include <stdio.h>
#include <cmath>

#include "mtmetis.h"
#include "metis.h"
#include "pulp.h"
#include "libmtkahypar.h"

VertexCut::~VertexCut() {
  if (partition != nullptr) delete[] partition;
}

VertexCut *VertexCut::Partition(CSR &csr, string partitioner, string params,
                                string order_method, int np) {
  VertexCut *vc = new VertexCut;

  cout << "Ordering..." << endl;
  vc->order = gen_order(csr.row_ptr, csr.col, csr.n, csr.m, order_method);

  cout << "Ranking..." << endl;
  vc->ranks.resize(csr.n);

  double part_start = omp_get_wtime(); 

#pragma omp parallel for num_threads(NUM_THREADS)
  for (IDType i = 0; i < csr.n; i++) {
    vc->ranks[vc->order[i]] = i;
  }

#pragma omp parallel for num_threads(NUM_THREADS)
  for(IDType u = 0; u < csr.n; u++){
    IDType start = csr.row_ptr[u];
    IDType end = csr.row_ptr[u+1];

    sort(csr.col + start, csr.col + end, [vc](IDType x, IDType y){
      return vc->ranks[x] < vc->ranks[y];
    });
  }

  int* weights = new int[csr.n];
  // 0 -> uniform
  // 1 -> degree
  // 2 -> degree_log
  int weight_strat = 0;

  if(PART_WEIGHTS == "uniform"){
    weight_strat = 0;
  } else if(PART_WEIGHTS == "degree"){
    weight_strat = 1;
  } else {
    weight_strat = 2;
  }

  size_t weight_sum = 0;
#pragma omp parallel for num_threads(NUM_THREADS)
  for (IDType u = 0; u < csr.n; u++) {

    IDType start = csr.row_ptr[u];
    IDType end = csr.row_ptr[u+1];

    int32_t weight = (weight_strat == 2) ? std::log(end - start) + 1 : (weight_strat == 1) ? end - start + 1 : 1;
    if constexpr(PART_LB_OFFSET != -1) weight += PART_LB_OFFSET;
    
    // Stranded or Leaf or Local Minimum
    if constexpr(PART_LB_OFFSET != -1)
      if(start == end || start + 1 == end || vc->ranks[u] < vc->ranks[csr.col[start]]){
        weight = 1;
      } 
    
    weights[u] = weight;
    weight_sum += weight;

  }


  if (partitioner == "pulp") {

    long* row_ptr_long = new long[csr.n+1];

#pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t i=0; i<csr.n+1; i++){
      row_ptr_long[i] = (long) csr.row_ptr[i];
    }

    int* edge_weights = new int[csr.m];
    fill(edge_weights, edge_weights + csr.m, 1); 

    pulp_graph_t graph;
    graph.n = csr.n;
    graph.m = csr.m;
    graph.out_array = csr.col;
    graph.out_degree_list = row_ptr_long;
    graph.vertex_weights = weights;
    graph.edge_weights = edge_weights;
    graph.vertex_weights_sum = (long) weight_sum;

    pulp_part_control_t con;
    con.vert_balance = 1.10;
    con.edge_balance = 1.5;
    con.pulp_seed = 42;
    con.do_lp_init = false;
    con.do_bfs_init = true;
    con.do_repart = true;
    con.do_edge_balance = true;
    con.do_maxcut_balance = true;

    int *partition = new int[csr.n];
    pulp_run(&graph, &con, partition, np);

    vc->partition = partition;
    vc->Init(csr, np);

  } else if(partitioner == "metis"){
    idx_t options[METIS_NOPTIONS];
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
    options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
    options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
    options[METIS_OPTION_NO2HOP] = 0;
    options[METIS_OPTION_NCUTS] = 1;
    options[METIS_OPTION_NITER] = 10;
    options[METIS_OPTION_UFACTOR] = 30;
    options[METIS_OPTION_MINCONN] = 0;
    options[METIS_OPTION_CONTIG] = 0;
    options[METIS_OPTION_SEED] = 42;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_DBGLVL] = 0;

    int *partition = new int[csr.n];

    int nw = 1;
    int objval;
    METIS_PartGraphKway(&csr.n, &nw, (idx_t *)csr.row_ptr,
                            (idx_t *)csr.col, weights, nullptr, nullptr,
                            &np, nullptr, nullptr, options, &objval, partition);

    vc->partition = partition;
    vc->Init(csr, np);

  } else if(partitioner == "mtkahypar"){

#ifdef ENABLE_MT_KAHYPAR
    mt_kahypar_initialize_thread_pool(NUM_THREADS, true);
    mt_kahypar_context_t* context = mt_kahypar_context_new();
    params = (params.empty()) ? "mtkahypar_config/quality_preset.ini" : params;
    mt_kahypar_configure_context_from_file(context, params.c_str());
    mt_kahypar_hypernode_id_t num_vertices = csr.n;
    mt_kahypar_hypernode_id_t num_hyperedges = csr.m;

    std::unique_ptr<mt_kahypar_hyperedge_weight_t[]> hyperedge_weights =
      std::make_unique<mt_kahypar_hyperedge_weight_t[]>(csr.m);
    #pragma parallel for num_threads(NUM_THREADS)
    for(size_t i=0; i<csr.m; i++){
      hyperedge_weights[i] = 1;
    }

    std::unique_ptr<mt_kahypar_hypernode_weight_t[]> vertex_weights  =
      std::make_unique<mt_kahypar_hypernode_weight_t[]>(csr.n);
    #pragma parallel for num_threads(NUM_THREADS)
    for(size_t i=0; i<csr.n; i++){
      vertex_weights[i] = weights[i];
    }

    std::unique_ptr<size_t[]> hyperedge_indices = std::make_unique<size_t[]>(csr.m+1);
    #pragma parallel for num_threads(NUM_THREADS)
    for(IDType i=0; i<csr.m+1; i++){
      hyperedge_indices[i] = 2*i;
    }

    std::unique_ptr<mt_kahypar_hyperedge_id_t[]> hyperedges = std::make_unique<mt_kahypar_hyperedge_id_t[]>(csr.m*2);
    size_t index = 0;
    for(IDType u=0; u<csr.n; u++){
      IDType ngh_start = csr.row_ptr[u];
      IDType ngh_end = csr.row_ptr[u+1];

      for(IDType i=ngh_start; i<ngh_end; i++){
        IDType v = csr.col[i];

        hyperedges[index++] = u;
        hyperedges[index++] = v;
      }
    }


    const double imbalance = 0.03;
    const mt_kahypar_partition_id_t k = np;

    mt_kahypar_hyperedge_weight_t objective = 0;

    std::vector<mt_kahypar_partition_id_t> partition(num_vertices, -1);

    mt_kahypar_partition(num_vertices, num_hyperedges,
                        imbalance, k, 42,
                        vertex_weights.get() , hyperedge_weights.get(),
                        hyperedge_indices.get(), hyperedges.get(),
                        &objective, context, partition.data(),
                        false);

    vc->partition = new int[csr.n];
    copy(partition.begin(), partition.end(), vc->partition);
    vc->Init(csr, np);

#else
    cerr << "MTKahypar was not linked, please recompile with ENABLE_MT_KAHYPAR=true or try a different partitioner" << endl;
    throw 1;
#endif

  } else if(partitioner == "mtmetis") {
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

      double *options = mtmetis_init_options();
      options[MTMETIS_OPTION_NTHREADS] = (double)NUM_THREADS;
      options[MTMETIS_OPTION_NRUNS] = (double)8;
      options[MTMETIS_OPTION_RUNSTATS] = (double)1;
      options[MTMETIS_OPTION_UBFACTOR] = (double) 1.03;

      double part_time = omp_get_wtime();
      MTMETIS_PartGraphKway(&nvtxs, &ncon, xadj, adj, weights, &vsize, adjwgt,
                          &nparts, tpwgts, &ubvec, options, &r_edgecut, where);
      cout << "Partition Time: " << omp_get_wtime() - part_time << " seconds"
         << endl;

      vc->partition = (IDType *)where;

      vc->Init(csr, np);
  } else {
    cerr << "Partitioner " << partitioner << " is not supported ! (please see the readme file for the available ones)" << endl;
    throw 1;
  }

  double part_end = omp_get_wtime(); 

  cout << "Partition Time: " << part_end - part_start << endl;

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
