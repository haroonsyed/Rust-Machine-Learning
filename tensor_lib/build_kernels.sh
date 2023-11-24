source ./.venv/bin/activate
cd ./cuda_kernels
rm -r build
mkdir build
cd ./build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
cd ..