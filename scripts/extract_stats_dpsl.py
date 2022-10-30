from audioop import avg
import sys
import statistics as stat

filename = sys.argv[1]

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

def process_stats(stats, graph, run_count):
    s = stats[graph]
    mean_time = stat.mean(s["time"])
    std_time = stat.stdev(s["time"]) if len(s["time"]) > 1 else 0
    mean_merge = stat.mean(s["merge"])
    mean_part = stat.mean(s["part"])
    max_mem = s["max_mem"]
    tot_mem = s["tot_mem"] / run_count

    return [mean_time, std_time, mean_merge, mean_part, max_mem, tot_mem]

# graph_name -> stat_name -> list of values
stats = {}
graph = ""
run = ""
run_count = 0
with open(filename, "r") as file:
    for line in file.readlines():
        if "__" in line:
            if "__END__" in line:
                pass
            else:
                graph, run = line.replace("\n", "").replace("__", "").split(":")
                if int(run) == 1:
                    stats[graph] = {"time" : [], "max_mem" : 0, "tot_mem": 0, "merge" : [], "part": []}
        else:
            if "P0: Total," in line:
                stats[graph]["time"].append(get_number(line))
                run_count = max(len(stats[graph]["time"]), run_count)
            elif "Partition Time" in line:
                stats[graph]["part"].append(get_number(line))
            elif "P0: Total Merge Time" in line:
                stats[graph]["merge"].append(get_number(line))
            elif "Label Memory," in line:
                mem = get_number(line)
                stats[graph]["max_mem"] = max(stats[graph]["max_mem"], mem)
                stats[graph]["tot_mem"] += mem



print("graph,time,stdev_time,merge,part,max_mem,tot_mem")
for graph in stats:
    s = process_stats(stats, graph, run_count)
    s_str = [str(x) for x in s]
    print(graph + "," + ",".join(s_str))
