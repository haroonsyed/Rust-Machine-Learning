source ./.venv/bin/activate
cd ./cuda_kernels
mkdir build
cd ./build
cmake ..
make
cd ..
cd ..