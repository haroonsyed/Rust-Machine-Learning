cd ..
cd ..
./build_kernels.sh
cd ./cuda_kernels/benchmarks
gcc -O3 main.cpp -o main -lstdc++ -L .. -lcuda_kernels -L /usr/local/cuda/lib64 -lcublas -lcudart
./main