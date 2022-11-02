import sys

# (level, node) -> stat_name -> value
lstats = {}

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

with open(filename,"r") as file:
    for line in file.readlines():
        if "Level" in line:
            try:
                n = int(line[1])
            except:
                n = 0

            l = line.split()[2].replace(",","")
            if "P" not in line:
                l = line.split()[1].replace(",","")

            if l == "0&1":
                continue;
            
            
            l = int(l)-1 if "&" not in line else 1
       

            if (l,n) not in lstats:
                lstats[(l,n)] = {
                    "time" : 0,
                    "count" : 0,
                    "cut_count" : 0,
                }

            if "Count" in line:
                c1 = get_number(line.split("(")[0])
                c2 = get_number(line.split("(")[-1].replace(")", ""))
                lstats[(l,n)]["count"] = int(c1)
                lstats[(l,n)]["cut_count"] = int(c2)
            else:
                t = get_number(line)
                lstats[(l,n)]["time"] = t

keys = list(sorted(lstats.keys()))
max_l, max_n = keys[-1]

time_csv = "level," + ",".join(["P" + str(x) for x in range(0,max_n+1)]) + "\n" + "\n".join([str(j) + "," + ", ".join([str(lstats[(j,i)]["time"]) for i in range(0,max_n+1)]) for j in range(2, max_l+1)])
count_csv = "level," + ",".join(["P" + str(x) for x in range(0,max_n+1)]) + "\n" + "\n".join([str(j) + "," + ", ".join([str(lstats[(j,i)]["count"]) for i in range(0,max_n+1)]) for j in range(2, max_l+1)])
cut_count_csv = "level," + ",".join(["P" + str(x) for x in range(0,max_n+1)]) + "\n" + "\n".join([str(j) + "," + ", ".join([str(lstats[(j,i)]["cut_count"]) for i in range(0,max_n+1)]) for j in range(2, max_l+1)])

print("Time:")
print(time_csv)
print("")

print("Count:")
print(count_csv)
print("")

print("Cut Count:")
print(cut_count_csv)
print("")



