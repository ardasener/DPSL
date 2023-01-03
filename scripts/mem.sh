for GRAPH in FBA FBB POKE LIVE; do
    echo "__${GRAPH}__"
    mpirun -n 4 --bind-to none ./dpsl graphs/${GRAPH}.mtx mtkahypar mtkahypar_config/default_v0.ini
done
