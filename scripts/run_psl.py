from cmath import exp
import os
import subprocess as sp
import statistics as stat
import sys

# Usage: <script> <psl_exe>

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
    memory = -1
    for line in out.split("\n"):
        if "Indexing:" in line:
            time = get_number(line)
        elif "Label Memory" in line:
            memory = get_number(line)
    return time, memory

graphs = [os.path.join("graphs", file) for file in os.listdir("graphs") if ".mtx" in file]

n_runs = 3
timeout = 2000
psl_exe = sys.argv[1]

include = ["DELI", "LAST"]

def ShouldRun(graph):
    for i in include:
        if i in graph:
            return True
    return False


res = "\n\n\ngraph,avg_time,stdev_time,mem\n"
for graph in graphs:

    if not ShouldRun(graph):
        continue;

    name = graph.split("/")[-1].replace(".mtx", "")

    error = False
    times = []
    memory = []
    print("Running:", graph)
    for i in range(0,n_runs):
        try:
            out = sp.check_output([psl_exe, graph], timeout=timeout).decode("utf-8")
            print(out)
            time, mem = get_stats(out)
            times.append(time)
            memory.append(mem)
        except sp.CalledProcessError as ex:
            error = True
            print("Called Process Error: ", ex.cmd, ex.returncode)
            print(ex.output)
        except Exception as ex:
            error = True
            print("Exception:", ex)
            res += name + "," + "INF" + "," + "INF" + "," + "INF" + "\n"
            break

    if not error:
        res += name + "," + str(sum(times) / n_runs) + "," + str(stat.stdev(times)) + "," + str(sum(memory) / n_runs) + "\n"

print(res)