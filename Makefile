release:
	g++ src/main.cpp -O3 -fopenmp -std=c++17
debug:
	g++ src/main.cpp -O0 -g -fopenmp -std=c++17