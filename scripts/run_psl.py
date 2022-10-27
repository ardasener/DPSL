from cmath import exp
import os
import subprocess as sp

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

log = open("run.log", "w+")
res = open("run.res", "w+")

n_runs = 3
timeout = 2000

include = ["FLIX", "CITE", "DBLP", "TOPC"]

def ShouldRun(graph):
    for i in include:
        if i in graph:
            return True
    return False


for graph in graphs:

    if not ShouldRun(graph):
        continue;

    tot_time = 0
    tot_memory = 0
    print("Running:", graph)
    log.write("Running: " + graph + "\n")
    for i in range(0,n_runs):
        try:
            out = sp.check_output(["./psl", graph], timeout=timeout).decode("utf-8")
            log.write(out + "\n")
            time, mem = get_stats(out)
            tot_time += time
            tot_memory += mem
        except Exception as ex:
            print("Exception:", ex)
            log.write("Exception: ", ex)
            res.write(graph + "," + "INF" + "," + "INF" + "\n")
            break
    res.write(graph + "," + str(tot_time / n_runs) + "," + str(tot_memory / n_runs) + "\n")