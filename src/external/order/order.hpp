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
vector<T> gen_order(T *xadj, T *adj, T n, T m, string method); 

// Makes it header only
#include "order.cpp"

#endif
