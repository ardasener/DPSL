#!/usr/bin/env python3

# Usage: <exe> <tests> <graphs> <max_proc_count>
# <tests>: comp, bp, order, ufactor, 
# <graphs>: names of graphs in the graphs folder
# <max_proc_count> : maximum number of processes used by mpi

import os
import sys
import subprocess as sp
from datetime import datetime
import itertools

def GenerateConfigs(tests_arg, graphs_arg):
    graphs = [os.path.join("part_graphs", graph) for graph in os.listdir("graphs") if graph in graphs_arg]
    comp_levels = [0, 1, 2, 3] if "comp" in tests_arg else [3] 
    bp_roots = [0, 15, 30, 45, 60] if "bp" in tests_arg else [15]
    order_methods = ["degree", "b_cent", "c_cent", "eigen_cent", "degree_b_cent", "degree_c_cent", "degree_eigen_cent"] if "order" in tests_arg else ["degree_eigen_cent"]
    ufactor=[1.03, 1.05, 1.1, 1.2] if "ufactor" in tests_arg else [1.03]
    return list(itertools.product(graphs, comp_levels, bp_roots, order_methods, ufactor))

def CreateOutputFolder():
    host = os.uname()[1]
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    folder = os.path.join("runs", date + "-" + host)
    sp.run(["mkdir", "-p", folder])
    return folder

def Execute()

tests_arg = sys.argv[1];
graphs_arg = sys.argv[2];
max_proc_count = sys.argv[3];

configs = GenerateConfigs(tests_arg, graphs_arg)
out_folder = CreateOutputFolder()

for i in range(2, max_proc_count+1, )