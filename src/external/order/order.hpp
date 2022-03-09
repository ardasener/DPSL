#pragma once

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

tuple<vector<int>,int*,int*,int,int> gen_order(int *xadj, int *adj, int n, int m, string method); 


// Makes it header only
#include "order.cpp"
