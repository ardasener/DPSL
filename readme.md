# DPSL

DPSL is a distributed partitioning based implementation of the Parallel Short-distance Labelling (PSL) algorithm. DPSL can run on multiple nodes distributing the graph and labels among the nodes. This results in both faster indexing times by allowing the use of more computation resources and less storage requirements for each node since the resulting labels will be distributed among them with minimal duplication.



# Building & Running

## Requirements

- GNU Make (tested with version 4.2.1)
- g++ (tested with versions 7.0.0 and 9.3.0)
- (For DPSL and GPSL only) openmpi (with `mpic++` compiler and `mpirun` tool) (tested with version 3.0.0)
- (For GPSL only) Nvidia CUDA (with `nvcc` compiler) (tested with version 11.2)

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
- `-n <node_count>`: Node count is the number of nodes we want to run the program on. It should much the number of partitions on the `<part_file>`.
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
- `<partitioner>`: A partitioner binary. Currently `gpmetis`, `pulp` and `mtmetis` are supported.
- `<partitioner_params>` : Parameters to be passed to the partitioner. (Please surround these with quotes to ensure they are passed as a single argument)