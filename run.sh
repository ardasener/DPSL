#! /bin/bash

# Usage: run.sh mode graphs
# mode: schedule, rank or none
# graphs: comma-seperated names of graphs
MODE=$1

if [ -d $2 ]; then
	GRAPHS=($(basename -a ${2}/*.mtx)) 
else
	GRAPHS=($(echo $2 | tr "," "\n"))
fi

echo "Running with: $GRAPHS"

mkdir -p run_outputs

function run_psl {
	./psl $1 | tee $2
}

function run_dpsl {
	mpirun -np 2 --bind-to none ./dpsl $1 $2 | tee $3
}

for GRAPH in ${GRAPHS[@]}; do

	if [ -d $2 ]; then
		GRAPH_PATH=${2}/${GRAPH}
	else
		GRAPH_PATH=part_graphs/${GRAPH}.mtx
		GRAPH=${GRAPH}.mtx
	fi

	VSEP_UNIFORM_PATH=vsep_output/${GRAPH}.uniform.vsep

	if [[ $MODE == "schedule" ]]; then
		for METHOD in static dynamic guided; do
			for CHUNK_SIZE in 1 2 4 8 16 32 64 128 256 512; do
				export OMP_SCHEDULE="${METHOD},${CHUNK_SIZE}"
				SCHED_INFO="${METHOD}_${CHUNK_SIZE}"
			
				echo "__PSL Schedule=${SCHED_INFO}__"	
				run_psl $GRAPH_PATH run_outputs/${GRAPH}.${SCHED_INFO}.psl.out
				echo "__DPSL:UNIFORM Schedule=${SCHED_INFO}__"	
				run_dpsl $GRAPH_PATH $VSEP_UNIFORM_PATH run_outputs/${GRAPH}.${SCHED_INFO}.dpsl.uniform.out
			done
		done
	fi

	
	if [[ $MODE == "rank" ]]; then
		for RERANK in "true" "false"
		do
			for GLOBAL_RANKS in "true" "false"
			do
				make psl -B RERANK=$RERANK GLOBAL_RANKS=$GLOBAL_RANKS
				make dpsl -B RERANK=$RERANK GLOBAL_RANKS=$GLOBAL_RANKS
				
				RANKS_INFO="${RERANK}_${GLOBAL_RANKS}"
				echo "__PSL Ranks=${RANKS_INFO}__"
				run_psl $GRAPH_PATH run_outputs/${GRAPH}.${RANKS_INFO}.psl.out
				echo "__DPSL:UNIFORM Ranks=${RANKS_INFO}__"	
				run_dpsl $GRAPH_PATH $VSEP_UNIFORM_PATH run_outputs/${GRAPH}.${RANKS_INFO}.dpsl.uniform.out
			done
		done	
	fi

	if [[ $MODE == "order_method" ]]; then
		for METHOD in degree b_cent c_cent rw_cent eigen_cent degree_b_cent degree_c_cent degree_rw_cent degree_eigen_cent
		do
			make psl -B NUM_THREADS=20 ORDER_METHOD=${METHOD}
			run_psl $GRAPH_PATH run_outputs/${GRAPH}.${METHOD}.psl.out
		done
	fi

	if [[ $MODE == "opt_bp" ]]; then
		for N_ROOTS in 16 32 64 128 256 512 1024
		do
			echo "__PSL:BP${N_ROOTS}__"
			make psl -B NUM_THREADS=60 N_ROOTS=${N_ROOTS}
			run_psl $GRAPH_PATH run_outputs/${GRAPH}.bp_${N_ROOTS}.psl.out
		done
	fi
		
	if [[ $MODE == "none" ]]; then
			echo "__PSL__"
			run_psl $GRAPH_PATH run_outputs/${GRAPH}.psl.out

			for VSEP_TYPE in b_cent_log degree degree_log lf_cent_log random rw_cent rw_cent_log uniform 
			do
				echo "__DPSL:${VSEP_TYPE}__"	
				run_dpsl $GRAPH_PATH vsep_output/${GRAPH}.${VSEP_TYPE}.vsep run_outputs/${GRAPH}.dpsl.${VSEP_TYPE}.out
			done

	fi

done	
