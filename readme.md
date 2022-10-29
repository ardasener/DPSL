# DPSL

DPSL is a distributed partitioning based implementation of the Parallel Shortest-distance Labelling (PSL) algorithm. DPSL can run on multiple nodes distributing the graph and labels among the nodes. This results in both faster indexing times by allowing the use of more computation resources and less storage requirements for each node since the resulting labels will be distributed among them with minimal duplication.

> Disclamer: This work is designed specifically for social networks, web graphs and other low diameter graphs.
> It will not work well on road networks. 
> And in fact with certain features turned on, it might give incorrect results due to overflows on high diameter graphs.
> On such graphs, we suggest using "ORDER_METHOD=b_cent COMP_LVL=1 USE_BP=false".
> Note that there is a limit on the maximum distance.

# Building & Running

## Requirements

- GNU Make (tested with version 4.2.1)
- g++ (tested with versions 7.0.0 and 9.3.0)
- (For DPSL only) openmpi (with `mpic++` compiler and `mpirun` tool) (tested with version 3.0.0)
- (For DPSL only) mtmetis (.a and .h files)
- (For GPSL only) Nvidia CUDA (with `nvcc` compiler) (tested with version 11.2)

## MTMetis Dependency

For DPSL, we need the mtmetis partitioner available [here](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview).
On most Linux systems, you can use `scripts/get_mtmetis.py` script to obtain and build it automatically.

> Please run the script on the project root. Like: `python3 scripts/get_mtmetis.py`.

Otherwise you will need to build it manually and place the `libmtmetis.a` and `mtmetis.h` files in the `mtmetis` directory.

## Building

Basic syntax:

``` bash
make -B <binary> <options>
```

### Binaries

This project currently includes 3 different binaries that could be build:
- `dpsl` : The main result of the project, the distributed PSL implementation
- `psl` : A reimplementation of the original PSL algorithm. According to our tests this version is faster than the original. It also provides several quality-of-life features like being able to read different file formats.
- `gpsl` : An experimental GPU implementation (currently WIP)

### Options

Please see `Makefile` for the options. Note that options marked experimental are not guaranteed to work.

> Some of the options for ORDER_METHOD are estimated values and are calculated using a multi-threaded approach that may result in slighly different results from run to run.
> As a result certain metrics like memory usage may vary when they are used.
> Furthermore the graph is assumed to be connected for many of these calculations and the program may perform poorly if it is not.
> For precise measurements, we suggest sticking to "degree" ordering.

### Examples

- Building PSL with 16 threads and BP turned off:

``` bash
make -B psl NUM_THREADS=16 USE_BP=false
```

- Building DPSL with degree ordering and 32 BP roots

``` bash
make -B dpsl ORDER_METHOD=degree N_ROOTS=32
```

## Running

### PSL

The PSL binary takes a single argument which is the graph file to be used. The file could be in edge list or matrix market file formats.

``` bash
./psl <graph_file>
```

### DPSL

#### Usage 1

``` bash
mpirun --bind-to none -n <node_count> ./dpsl <graph_file> <part_file>
```

- `--bind-to none`: This part ensures that all the cores on the node are available to the program.
- `-n <node_count>`: Node count is the number of nodes we want to run the program on. It should match the number of partitions on the `<part_file>`.
- `<graph_file>`: Same as PSL.
- `<part_file>`: Each line of this file should contain a single integer which should be the partition id for the corresponding vertex. Outputs from `metis` or `mtmetis` should work fine here. As mentioned before the number of partitions should match the number of nodes.

> Note that this might not work well with the COMPRESS option turned on, as the load balance will be thrown off due to the removal of some vertices.
> When used with the COMPRESS option we suggest Usage 2.

#### Usage 2

``` bash
mpirun --bind-to none -n <node_count> ./dpsl <graph_file> <partitioner> <partitioner_params>
```

- `--bind-to none`: This part ensures that all the cores on the node are available to the program.
- `-n <node_count>`: Node count is the number of nodes we want to run the program on. It should much the number of partitions on the `<part_file>`.
- `<graph_file>`: Same as PSL.
- `<partitioner>`: A partitioner name. Currently only "mtmetis" is supported.

# Datasets

We used data from [SuiteSparse](https://sparse.tamu.edu/) and [Network Repository](https://networkrepository.com/) for evaluation.
The `scripts/download.py` script can be used to automatically download our dataset on most Linux systems.

> Please run the script on the project root. Like: `python3 scripts/download.py`.