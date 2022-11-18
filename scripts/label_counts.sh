for COMP in 0 3; do 
    make psl -B NUM_THREADS=60 COMP_LVL=${COMP} MODE=FAST_DEBUG
    for GRAPH in DELI LAST DIGG CITE DBLP TOPC FLIX; do
    # for GRAPH in DELI; do
        ./psl graphs/${GRAPH}.mtx
        cp output_psl_label_counts.txt ${GRAPH}_lc_C${COMP}.txt
        cp output_psl_cand_counts.txt ${GRAPH}_cc_C${COMP}.txt
    done
done

mkdir -p lc
mkdir -p cc
mv *_lc_*.txt lc/
mv *_cc_*.txt cc/
zip -r lc.zip lc/
zip -r cc.zip cc/
rm -rf lc/ cc/