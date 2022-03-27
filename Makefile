release:
	mpic++ src/main.cpp -O3 -fopenmp -std=c++17 -L. libmetis.so -Wl,-rpath=.
debug:
	mpic++ src/main.cpp -O0 -g -fopenmp -std=c++17 -L. libmetis.so -Wl,-rpath=. -DDEBUG
