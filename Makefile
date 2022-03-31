release:
	mpic++ src/main.cpp -O3 -fopenmp -std=c++17 -l:libmetis.a -l:libpatoh.a -l:libkahypar.so -Wl,-rpath /usr/local/lib -no-pie
debug:
	mpic++ src/main.cpp -O0 -g -fopenmp -std=c++17 -l:libmetis.a -l:libpatoh.a -l:libkahypar.so -Wl,-rpath /usr/local/lib -no-pie -DDEBUG
