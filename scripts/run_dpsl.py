from cmath import exp
import os
import subprocess as sp
import sys
import statistics as stat

# Usage: <script> <dpsl_exe> <np> <partitioner_name>

def get_number(s):
    s = s.replace("\n", "")
    res = -1
    res_len = 0
    for w in s.split(" "):
        try:
            f = float(w)
            if len(w) > res_len:
                res_len = len(w)
                res = f
        except:
            continue
    return res

def get_stats(out):
    time = -1
    merge = -1
    max_mem = -1
    tot_mem = -1
    part_time = -1
    for line in out.split("\n"):
        if "Total, " in line:
            time = max(get_number(line), time)
        elif "Total Merge Time," in line:
            merge = max(get_number(line), merge)
        elif "Partition Time:" in line:
            part_time = get_number(line)
        elif "Label Memory," in line:
            if tot_mem == -1:
                tot_mem = 0
            mem = get_number(line)
            tot_mem += mem
            max_mem = max(mem, max_mem)
    return time, merge, max_mem, tot_mem, part_time

graphs = [os.path.join("graphs", file) for file in os.listdir("graphs") if ".mtx" in file]

n_runs = 3
timeout = 2000

dpsl_exe = sys.argv[1]
np = sys.argv[2]
part = sys.argv[3]

include = ["DELI", "LAST"]

def ShouldRun(graph):
    for i in include:
        if i in graph:
            return True
    return False

res = "\n\n\ngraph,avg_time,stdev_time,avg_merge_time,part_time,max_mem,tot_mem\n"
for graph in graphs:

    if not ShouldRun(graph):
        continue;

    name = graph.split("/")[-1].replace(".mtx", "")

    error = False

    times = []
    merge_times = []
    part_times = []
    tot_memory = []
    max_memory = []
    print("Running:", graph)
    for i in range(0, n_runs):
        try:
            out = sp.check_output(["mpirun", "-n", np, "--bind-to", "none", dpsl_exe, graph, part], timeout=timeout).decode("utf-8")
            print(out)
            time, merge, max_mem, tot_mem, part_time = get_stats(out)
            part_times.append(part_time)
            times.append(time)
            merge_times.append(merge)
            tot_memory.append(tot_mem)
            max_memory.append(max_mem)
        except Exception as ex:
            print("Exception:", ex)
            error = True
            res += name + "," + "INF" + "," + "INF" + "," + "INF" + "," + "INF" + "," + "INF" + "\n"
            break

    if not error:
        res += name + "," + str(sum(times) / n_runs) + "," \
            + str(stat.stdev(times)) + "," \
            + str(sum(merge_times) / n_runs)  + "," \
            + str(sum(part_times) / n_runs)  + "," \
            + str(sum(max_memory) / n_runs) + "," \
            + str(sum(tot_memory) / n_runs) + "\n"    

print(res)