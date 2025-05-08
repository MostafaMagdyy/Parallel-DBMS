export PATH=/usr/local/cuda/bin:$PATH
DUCKDB_INCLUDE="./duckdb/src/include"
DUCKDB_LIB="./duckdb/build/release/src" 

# Compile CUDA files first
# nvcc -std=c++17 -arch=sm_86 -c  -o cuda/aggregate.o \


# Then compile C++ files and link everything
nvcc -std=c++17 -arch=sm_86 \
  cuda/aggregate_all.cu  -o aggregate_test \
  -I$DUCKDB_INCLUDE -I. \
  -L$DUCKDB_LIB -lduckdb -lcudart \
  -Xlinker -rpath=$DUCKDB_LIB \
  -rdc=true

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
else
    echo "Compilation successful!"
    # export LD_LIBRARY_PATH=$DUCKDB_LIB:$LD_LIBRARY_PATH
    # ./test_duckdb
    # echo "Execution completed with exit code $?"
fi