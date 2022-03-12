release:
	g++ src/main.cpp -O3 -fopenmp -std=c++17 -lmetis
debug:
	g++ src/main.cpp -O0 -g -fopenmp -std=c++17 -lmetis