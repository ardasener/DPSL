import subprocess as sp

link = "http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/mt-metis-0.7.2.tar.gz"

sp.run(["mkdir", "-p", "temp"])
sp.run(["mkdir", "-p", "mtmetis"])
sp.run(["wget", "-c", link], cwd="./temp")
sp.run(["tar", "xzf", "mt-metis-0.7.2.tar.gz"], cwd="./temp")
sp.run(["./configure"], cwd="./temp/mt-metis-0.7.2")
sp.run(["make"], cwd="./temp/mt-metis-0.7.2")
sp.run(["mv", "./temp/mt-metis-0.7.2/build/Linux-x86_64/lib/libmtmetis.a", "mtmetis/"])
sp.run(["mv", "./temp/mt-metis-0.7.2/include/mtmetis.h", "mtmetis/"])
sp.run(["rm", "-rf", "temp"])