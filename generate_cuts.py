import toml
import os
import subprocess as sp
import time
from multiprocessing import Pool

os.environ['GIS_LOCK'] = 'default'

ignore_existing = False

config = toml.load("config_dpsl.toml")

ufactors = [30, 60, 90, 120];
weights = ["uniform", "degree", "degree_log"]
graphs = [graph for graph in os.listdir("large_graphs") if ".part" not in graph]


config["partition_engine"] = "metis"

def already_exists(graph, ufactor, weight):
    for file in os.listdir("large_graphs"):
        if ".part" in file and graph in file and weight in file and str(ufactor) in file:
            return True
    return False

def process(graph):

    for ufactor in ufactors:
        for weight in weights:

            setting = str([graph, ufactor, weight])
            
            if (not ignore_existing) and already_exists(graph, ufactor, weight):
                print([graph, ufactor, weight, 0, "SKIPPED"])
                continue

            graph_path = os.path.join("large_graphs", graph)
            config["partition_weight"] = weight
            config["metis"]["ufactor"] = ufactor
            toml_string = toml.dumps(config)
            new_config_file = open("config_temp_" + graph.split(".")[0] + ".toml", "w+");
            new_config_file.write(toml_string)
            new_config_file.close()

            start = time.time()
            cmd = ["./cutter", "config_temp_" + graph.split(".")[0] + ".toml", graph_path]
            p = sp.Popen(cmd, shell=False, stdout=sp.PIPE, stderr=sp.PIPE)
            p.wait();
            end = time.time()

            if p.returncode != 0:
                print([graph, ufactor, weight, end-start, "FAILED: " + str(p.returncode)])
                # print(p.stderr.read())
                # print(p.stdout.read())
                continue
            else:
                print([graph, ufactor, weight, end-start, "DONE"])
                p = sp.Popen(["mv", graph_path + ".part2", graph_path + ".part2" + "." + weight + "_" + str(ufactor)], 
                        stdout=sp.PIPE, stderr=sp.PIPE, shell=False)
                p.wait()

                if p.returncode != 0:
                    print("Rename Error!")

                p = sp.Popen(["mv", graph_path + ".part4", graph_path + ".part4" + "." + weight + "_" + str(ufactor)],
                        stdout=sp.PIPE, stderr=sp.PIPE, shell=False)
                p.wait()

                if p.returncode != 0:
                    print("Rename Error!")



p = sp.Popen("rm " +  "./config_temp*.toml", shell=True)
p.wait()

# for graph in graphs:
#     process(graph)

with Pool(32) as pool:
    pool.map(process, graphs)

p = sp.Popen("rm " +  "./config_temp*.toml", shell=True)
p.wait()
