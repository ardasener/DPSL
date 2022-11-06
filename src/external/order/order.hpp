#ifndef ORDER_HPP
#define ORDER_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <random>
#include <regex>
#include <string>
#include <utility>
#include <tuple>

using namespace std;

template <typename T>
// Supported Methods:
// degree
// b_cent
// c_cent
// eigen_cent
// rw_cent
// degree_b_cent
// degree_c_cent
// degree_eigen_cent
// degree_rw_cent
vector<T> gen_order(T *xadj, T *adj, T n, T m, string method, double shuffle=0); 

// Makes it header only
#include "order.cpp"

#endif
