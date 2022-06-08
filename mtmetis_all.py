import subprocess as sp
import os

for graph in os.listdir("large_graphs"):
    if ".mtx" in graph:
        graph_path = os.path.join("large_graphs", graph)
        graph_part2 = graph_path + ".vsep"
        cmd = ["./mtmetis", "--ptype=vsep", graph_path, graph_part2]
        
        print("Partitioning: ", graph)
        sp.run(cmd)

