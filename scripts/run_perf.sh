mkdir -p perf_output

for NT in 15 30 60; do
    for BP in true false; do
        make psl -B NUM_THREADS=$NT COMP_LVL=3 USE_BP=$BP
        for GRAPH in DBLP TOPC FLIX; do
            echo "__${GRAPH}_${NT}_${BP}__"
            perf stat -d -o perf_output/${GRAPH}_${NT}_${BP}.perf ./psl graphs/$GRAPH.mtx
        done
    done
done