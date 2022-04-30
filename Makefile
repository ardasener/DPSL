release:
	mpic++ src/main.cpp -o dpsl -O3 -fopenmp -std=c++17 -L./libs -l:libmetis.so  -l:libkahypar.so -Wl,-rpath ./libs -no-pie
slow-release:
	mpic++ src/main.cpp -o dpsl -O0 -g -fopenmp -std=c++17 -L./libs -l:libmetis.so  -l:libkahypar.so -Wl,-rpath ./libs -no-pie
profile:
	mpic++ src/main.cpp -o dpsl -O3 -pg -fopenmp -std=c++17 -L./libs -l:libmetis.so  -l:libkahypar.so -Wl,-rpath ./libs -no-pie -DDEBUG
debug:
	mpic++ src/main.cpp -o dpsl -O0 -g -fopenmp -std=c++17 -L./libs -l:libmetis.so  -l:libkahypar.so -Wl,-rpath ./libs -no-pie -DDEBUG
fast-debug:
	mpic++ src/main.cpp -o dpsl -O3 -g -fopenmp -std=c++17 -L./libs -l:libmetis.so  -l:libkahypar.so -Wl,-rpath ./libs -no-pie -DDEBUG
fast-silent-debug:
	mpic++ src/main.cpp -o dpsl -O3 -g -fopenmp -std=c++17 -L./libs -l:libmetis.so  -l:libkahypar.so -Wl,-rpath ./libs -no-pie
cutter:
	g++ src/cutter.cpp -o cutter -O3 -g -fopenmp -std=c++17 -L./libs -l:libmetis.so  -l:libkahypar.so -Wl,-rpath ./libs -no-pie
gpu:
	nvcc src/gpsl.cu -o gpsl -O3 -std=c++17 -Xcompiler -fopenmp -Xcompiler -L./libs -Xcompiler -l:libmetis.so -Xcompiler -l:libkahypar.so -Xcompiler -no-pie -arch=sm_60
