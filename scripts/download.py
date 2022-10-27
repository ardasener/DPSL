from urllib.request import urlretrieve as urlret
import subprocess as sp
import os

links = """
DELI,https://nrvis.com/download/data/soc/soc-delicious.zip
LAST,https://nrvis.com/download/data/soc/soc-lastfm.zip
DIGG,https://nrvis.com/download/data/soc/soc-digg.zip
FLIX,https://nrvis.com/download/data/soc/soc-flixster.zip
CITE,https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersCiteseer.tar.gz
DBLP,https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/coPapersDBLP.tar.gz
TOPC,https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-topcats.tar.gz
FBA,https://nrvis.com/download/data/socfb/socfb-A-anon.zip
FBB,https://nrvis.com/download/data/socfb/socfb-B-anon.zip
POKE,https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Pokec.tar.gz
LIVE,https://suitesparse-collection-website.herokuapp.com/MM/LAW/ljournal-2008.tar.gz
SKIT,https://suitesparse-collection-website.herokuapp.com/MM/SNAP/as-Skitter.tar.gz
INDO,https://suitesparse-collection-website.herokuapp.com/MM/LAW/indochina-2004.tar.gz
RCAL,https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz
RTEX,https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-TX.tar.gz
"""

def FindMtx(dir):
    mtx_files = []
    for file in os.listdir(dir):
        if ".mtx" in file:
            mtx_files.append(file)
    return min(mtx_files, key=lambda x : len(x))


def SuiteSparseDownloader(name, link):
    urlret(link, "graphs/temp.tar.gz")
    sp.run(["tar", "xzf", "graphs/temp.tar.gz", "--overwrite", "-C", "graphs"])
    dir = os.path.join("graphs", link.split("/")[-1].replace(".tar.gz", ""))
    mtx = os.path.join(dir, FindMtx(dir)) 
    new_mtx_path = os.path.join("graphs", name + ".mtx")
    sp.run(["mv", mtx, new_mtx_path])
    sp.run(["rm" +  " -rf" +  " graphs/*.tar.gz " +  dir], shell=True)

def NetRepoDownloader(name, link):
    urlret(link, "graphs/temp.zip")
    sp.run(["unzip","-o", "graphs/temp.zip", "-d", "graphs"])
    mtx_name = link.split("/")[-1].replace(".zip", ".mtx")
    mtx_path = os.path.join("graphs", mtx_name)
    new_mtx_path = os.path.join("graphs", name + ".mtx")
    sp.run(["mv", mtx_path, new_mtx_path])

    with open(new_mtx_path,'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("%" + content)
    
    sp.run("rm" + " -f" +  " graphs/readme*" + " graphs/*.zip", shell=True)


sp.run(["mkdir", "-p", "graphs"])

for line in links.split("\n"):

    if line == "":
        continue


    name, link = line.replace("\n","").split(",")

    if os.path.isfile(os.path.join("graphs", name + ".mtx")):
        print("Skipping:", name)
        continue


    print("Downloading:", name)

    if "suitesparse" in link:
        SuiteSparseDownloader(name, link)
    elif "nrvis" in link:
        NetRepoDownloader(name, link)
    else:
        print("Unknown link:", link)