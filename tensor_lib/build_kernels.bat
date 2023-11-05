call ".\.venv\Scripts\activate.bat"
cd "./cuda_kernels"
mkdir build
cd ./build
cmake ..
cmake --build . --config Release
cd ..
cd ..
