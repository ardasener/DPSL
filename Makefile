release:
	mpic++ src/main.cpp -O3 -fopenmp -std=c++17 -L./libs -l:libmetis.so -l:libpatoh.a -l:libkahypar.so -Wl,-rpath ./libs -no-pie
debug:
	mpic++ src/main.cpp -O0 -g -fopenmp -std=c++17 -L./libs -l:libmetis.so -l:libpatoh.a -l:libkahypar.so -Wl,-rpath ./libs -no-pie -DDEBUG
