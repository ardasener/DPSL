#ifndef ORDER_CPP
#define ORDER_CPP

#include "order.hpp"
#include <cfloat>

using namespace std;

bool sortedge(const pair<int, int> &a, const pair<int, int> &b) {
  if (a.first == b.first) {
    return (a.second < b.second);
  } else {
    return (a.first < b.first);
  }
}

void computeBCandCC(int *xadj, int *adj, int n, int noBFS, int **que,
                    int **level, int **pred, int **endpred, float **sigma,
                    float **delta, float **b_cent, float **c_cent,
                    int maxlevel) {
  int i, j, v, w, qeptr, cur, ptr;
  long int sum;
  int nthreads = omp_get_max_threads();

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int *pque = que[tid];
    int *plevel = level[tid];
    int *ppred = pred[tid];
    int *pendpred = endpred[tid];
    float *psigma = sigma[tid];
    float *pdelta = delta[tid];
    float *pb_cent = b_cent[tid];
    float *pc_cent = c_cent[tid];
    int pi;

    std::mt19937 mt(tid * 1111);
    std::uniform_int_distribution<int> dist(0, n - 1);

    for (pi = 0; pi < n; pi++) {
      pb_cent[pi] = 0.0f;
      pc_cent[pi] = 0;
    }

#pragma omp for schedule(dynamic) private(ptr, qeptr, sum, cur, v, w, j, i)
    for (int x = 0; x < noBFS; x++) {
      i = dist(mt);

      qeptr = 1;
      pque[0] = i;
      sum = 0;

      for (j = 0; j < n; j++)
        pendpred[j] = xadj[j];
      for (j = 0; j < n; j++)
        plevel[j] = -2;
      for (j = 0; j < n; j++)
        psigma[j] = 0;
      plevel[i] = 0;
      psigma[i] = 1;

      // step 1: build shortest path graph
      cur = 0;
      while (cur != qeptr) {
        v = pque[cur];
        pc_cent[v] += 1.0f / (plevel[v] + 1);
        for (ptr = xadj[v]; ptr < xadj[v + 1]; ptr++) {
          w = adj[ptr];
          if (plevel[w] < 0 && (maxlevel == -1 || plevel[v] < maxlevel)) {
            plevel[w] = plevel[v] + 1;
            sum += plevel[w];
            pque[qeptr++] = w;
          }

          if (plevel[w] == plevel[v] + 1) {
            psigma[w] += psigma[v];
          } else if (plevel[w] == plevel[v] - 1) {
            ppred[pendpred[v]++] = w;
          }
        }
        cur++;
      }

      // step 2: compute betweenness in parallel
      for (j = 0; j < n; j++)
        pdelta[j] = 0.0f;
      for (j = qeptr - 1; j > 0; j--) {
        w = pque[j];
        for (ptr = xadj[w]; ptr < pendpred[w]; ptr++) {
          v = ppred[ptr];
          pdelta[v] += (psigma[v] * (1 + pdelta[w])) / psigma[w];
        }
        pb_cent[w] += pdelta[w];
      }
    }
  }

  // step 3: reduce the parallel results
  float *pb_cent_src = b_cent[0];
  float *pc_cent_src = c_cent[0];
#pragma omp parallel for
  for (j = 0; j < n; j++) {
    float sum_bcent = 0, sum_ccent = 0;
    for (int i = 1; i < nthreads; i++) {
      sum_bcent += b_cent[i][j];
      sum_ccent += c_cent[i][j];
    }
    pb_cent_src[j] += sum_bcent;
    pc_cent_src[j] += sum_ccent;
  }
}

// this is just to avoid repetitive computations in passes
void computeRWprobs(int *xadj, int *adj, int n, float *probs) {
  int i, j, ptr;
#pragma omp parallel for private(ptr) schedule(dynamic, 64)
  for (i = 0; i < n; i++) {
    for (ptr = xadj[i]; ptr < xadj[i + 1]; ptr++) {
      j = adj[ptr];
      probs[ptr] = 1.0f / (xadj[j + 1] - xadj[j]);
    }
  }
}

void computeRWCent(int *xadj, int *adj, int n, float *probs, float *curr_vals,
                   float *next_vals, int iter) {
  int i, j, ptr;
#pragma omp parallel for
  for (j = 0; j < n; j++) {
    curr_vals[j] = 1.0f;
    next_vals[j] = 0.0f;
  }
  for (i = 0; i < iter; i++) {
#pragma omp parallel for private(ptr) schedule(dynamic, 64)
    for (j = 0; j < n; j++) {
      for (ptr = xadj[j]; ptr < xadj[j + 1]; ptr++) {
        next_vals[j] += probs[ptr] * curr_vals[adj[ptr]];
      }
    }
#pragma omp parallel for
    for (j = 0; j < n; j++) {
      curr_vals[j] = next_vals[j];
      next_vals[j] = 0;
    }
  }
}

void computeEVCent(int *xadj, int *adj, int n, float *probs, float *curr_vals,
                   float *next_vals, int iter) {
  int i, j, ptr;
  float sum;
#pragma omp parallel for
  for (j = 0; j < n; j++) {
    curr_vals[j] = 1.0f;
    next_vals[j] = 0.0f;
  }
  for (i = 0; i < iter; i++) {
#pragma omp parallel for private(ptr) schedule(dynamic, 64)
    for (j = 0; j < n; j++) {
      for (ptr = xadj[j]; ptr < xadj[j + 1]; ptr++) {
        next_vals[j] += probs[ptr] * curr_vals[adj[ptr]];
      }
    }
    // find max in the vector
    sum = .0f;
    for (j = 0; j < n; j++)
      sum += curr_vals[j];
    sum = sqrt(sum);

#pragma omp parallel for
    for (j = 0; j < n; j++) {
      curr_vals[j] = next_vals[j] / sum;
      next_vals[j] = 0;
    }
  }
}

float pearson(float *x, float *y, int n) {
  int i;
  float ex, ey, xt, yt, sxx, syy, sxy;

  ex = ey = 0;
  for (i = 0; i < n; i++) {
    ex += x[i];
    ey += y[i];
  }
  ex /= n;
  ey /= n;

  sxx = syy = sxy = 0;
  for (i = 0; i < n; i++) {
    xt = x[i] - ex;
    yt = y[i] - ey;
    sxx += xt * xt;
    syy += yt * yt;
    sxy += xt * yt;
  }
  return sxy / (sqrt(sxx * syy));
}

vector<int> gen_order(int *xadj, int *adj, int n, int m, string method) {

  cout << "Generating order..." << endl;
  int max_deg = 0, min_deg = n, deg, degs[n];
  memset(degs, 0, sizeof(int) * n);

  for (int u = 0; u < n; u++) {
    deg = (xadj[u + 1] - xadj[u]);
    degs[deg]++;
    if (deg < min_deg) {
      min_deg = deg;
    }
    if (deg > max_deg) {
      max_deg = deg;
    }
  }

  cout << "---------------------------" << endl;
  cout << "No vertices is " << n << endl;
  cout << "No edges is " << m << endl;
  cout << "---------------------------" << endl;
  cout << "Min deg: " << min_deg << endl;
  cout << "Max deg: " << max_deg << endl;
  cout << "Avg deg: " << ((float)m) / n << endl;
  cout << "---------------------------" << endl;
  cout << "# deg 0: " << degs[0] << endl;
  cout << "# deg 1: " << degs[1] << endl;
  cout << "# deg 2: " << degs[2] << endl;
  cout << "# deg 3: " << degs[3] << endl;

  for (int i = n - 2; i > 0; i--) {
    degs[i] += degs[i + 1];
  }

  cout << "---------------------------" << endl;
  cout << "# deg>=32: " << degs[32] << endl;
  cout << "# deg>=64: " << degs[64] << endl;
  cout << "# deg>=128: " << degs[128] << endl;
  cout << "# deg>=256: " << degs[256] << endl;
  cout << "# deg>=512: " << degs[512] << endl;
  cout << "# deg>=1024: " << degs[1024] << endl;
  cout << "---------------------------" << endl << endl;

  int nthreads = omp_get_max_threads();
  cout << "Running with " << nthreads << " threads \n";
  int *pque[nthreads], *plevel[nthreads], *ppred[nthreads], *pendpred[nthreads];
  float *psigma[nthreads], *pdelta[nthreads], *pb_cent[nthreads],
      *pc_cent[nthreads];
  for (int i = 0; i < nthreads; i++) {
    pque[i] = new int[n];
    plevel[i] = new int[n];
    ppred[i] = new int[xadj[n]];
    pendpred[i] = new int[n];
    psigma[i] = new float[n];
    pdelta[i] = new float[n];
    pb_cent[i] = new float[n];
    pc_cent[i] = new float[n];
  }

  int *compid = new int[n];
  int *que = new int[n];

  int nocomp, qptr, qeptr, largestSize;
  nocomp = qptr = qeptr = largestSize = 0;

  for (int i = 0; i < n; i++) {
    compid[i] = -1;
  }

  int compsize, lcompid;
  for (int i = 0; i < n; i++) {
    if (compid[i] == -1) {
      compsize = 1;
      compid[i] = nocomp;
      que[qptr++] = i;

      while (qeptr < qptr) {
        int u = que[qeptr++];
        for (int p = xadj[u]; p < xadj[u + 1]; p++) {
          int v = adj[p];
          if (compid[v] == -1) {
            compid[v] = nocomp;
            que[qptr++] = v;
            compsize++;
          }
        }
      }
      if (largestSize < compsize) {
        lcompid = nocomp;
        largestSize = compsize;
      }
      nocomp++;
    }
  }

  int vcount, ecount;
  ecount = vcount = 0;
  for (int i = 0; i < n; i++) {
    if (compid[i] == lcompid) {
      que[i] = vcount++;
      for (int p = xadj[i]; p < xadj[i + 1]; p++) {
        if (compid[adj[p]] == lcompid)
          ecount++;
      }
    }
  }

  int *lxadj = new int[vcount + 1];
  int *ladj = new int[ecount];
  int *ltadj = new int[ecount];
  vcount = 0;
  ecount = 0;
  lxadj[0] = 0;
  for (int i = 0; i < n; i++) {
    if (compid[i] == lcompid) {
      vcount++;

      for (int p = xadj[i]; p < xadj[i + 1]; p++) {
        if (compid[adj[p]] == lcompid) {
          ladj[ecount++] = que[adj[p]];
        }
      }
      lxadj[vcount] = ecount;
    }
  }
  printf("largest component graph obtained with %d vertices %d edges -- %d\n",
         vcount, ecount, lxadj[vcount]);

  float *cent = new float[n];
  // float* one_btwn = new float[n];
  // float* two_btwn = new float[n];
  // float* thr_btwn = new float[n];
  // float* all_btwn = new float[n];
  // float* rw_cent = new float[n];
  // float* eigen_cent = new float[n];
  // float* c_cent = new float[n];
  float *probs = new float[xadj[n]];
  float *next_vals = new float[n];

  int noBFS = 256;
  int iter = 20;

  double start, end;

  if (method == "degree") {
    cout << "starting degree computation\n";
    start = omp_get_wtime();
    for (int i = 0; i < vcount; i++) {
      cent[i] = lxadj[i + 1] - lxadj[i];
    }
    end = omp_get_wtime();
    cout << "Deg-cent is computed in " << end - start << " seconds\n";
  }

  // cout << "starting BC-1 computation\n";
  // start = omp_get_wtime();
  // computeBCandCC(lxadj, ladj, vcount, noBFS, pque, plevel, ppred, pendpred,
  // psigma, pdelta, pb_cent, pc_cent, 2); end = omp_get_wtime(); for(int i = 0;
  // i < n; i++) one_btwn[i] = pb_cent[0][i]; cout << "BC-1 is computed in " <<
  // end - start << " seconds\n";

  // cout << "starting BC-2 computation\n";
  // start = omp_get_wtime();
  // computeBCandCC(lxadj, ladj, vcount, noBFS, pque, plevel, ppred, pendpred,
  // psigma, pdelta, pb_cent, pc_cent, 4); end = omp_get_wtime(); for(int i = 0;
  // i < n; i++) two_btwn[i] = pb_cent[0][i]; cout << "BC-2 is computed in " <<
  // end - start << " seconds\n";

  // cout << "starting BC-3 computation\n";
  // start = omp_get_wtime();
  // computeBCandCC(lxadj, ladj, vcount, noBFS, pque, plevel, ppred, pendpred,
  // psigma, pdelta, pb_cent, pc_cent, 6); end = omp_get_wtime(); for(int i = 0;
  // i < n; i++) thr_btwn[i] = pb_cent[0][i]; cout << "BC-3 is computed in " <<
  // end - start << " seconds\n";

  else if (method == "betweenness") {
    cout << "starting BC-all computation\n";
    start = omp_get_wtime();
    computeBCandCC(lxadj, ladj, vcount, noBFS, pque, plevel, ppred, pendpred,
                   psigma, pdelta, pb_cent, pc_cent, -1);
    end = omp_get_wtime();
    for (int i = 0; i < n; i++)
      cent[i] = pb_cent[0][i];
    cout << "BC-all is computed in " << end - start << " seconds\n";

  }

  else if (method == "closeness") {
    cout << "starting BC-all computation\n";
    start = omp_get_wtime();
    computeBCandCC(lxadj, ladj, vcount, noBFS, pque, plevel, ppred, pendpred,
                   psigma, pdelta, pb_cent, pc_cent, -1);
    end = omp_get_wtime();
    for (int i = 0; i < n; i++)
      cent[i] = pc_cent[0][i];
    cout << "BC-all is computed in " << end - start << " seconds\n";
  }

  else {
    // to compute edge probabilities
    computeRWprobs(xadj, adj, n, probs);

    if (method == "rw") {
      cout << "starting RW cent computation\n";
      start = omp_get_wtime();
      computeRWCent(lxadj, ladj, vcount, probs, cent, next_vals, iter);
      end = omp_get_wtime();
      cout << "RW cent is computed in " << end - start << " secs\n";
    } else if (method == "eigen") {
      cout << "starting EV cent computation\n";
      start = omp_get_wtime();
      computeEVCent(lxadj, ladj, vcount, probs, cent, next_vals, iter);
      end = omp_get_wtime();
      cout << "EV cent is computed in " << end - start << " secs\n";
    }
  }

  cout << "Ordering based on " << method << "..." << endl;
  vector<pair<float, int>> sort_vec;
  int li = 0;
  for (int i = 0; i < n; i++) {
    if(compid[i] == lcompid)
      sort_vec.emplace_back(cent[li++], i);
    else
      sort_vec.emplace_back(FLT_MIN,i);
  }
  sort(sort_vec.begin(), sort_vec.end(), less<pair<float, int>>());

  vector<int> order;
  for (auto const &p : sort_vec) {
    order.push_back(p.second);
  }

  for (int i = 0; i < nthreads; i++) {
    delete[] pque[i];
    delete[] plevel[i];
    delete[] ppred[i];
    delete[] pendpred[i];
    delete[] psigma[i];
    delete[] pdelta[i];
    delete[] pb_cent[i];
    delete[] pc_cent[i];
  }

  delete[] cent;
  delete[] probs;
  delete[] next_vals;

  delete[] lxadj;
  delete[] ladj;
  delete[] ltadj;

  return order;
}

#endif
