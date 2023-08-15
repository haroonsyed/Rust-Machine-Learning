source ./.venv/bin/activate
cd ./cuda_kernels
rm -r build
mkdir build
cd ./build
cmake ..
make
cd ..
cd ..