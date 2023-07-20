 gcc main.cpp -o main -lstdc++ -L .. -lcuda_kernels -L /usr/local/cuda/lib64 -lcublas -lcudart
 ./main