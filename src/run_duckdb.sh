export PATH=/usr/local/cuda/bin:$PATH
DUCKDB_INCLUDE="./duckdb/src/include"
DUCKDB_LIB="./duckdb/build/release/src" 

# Compile CUDA files first
nvcc -std=c++17 -arch=sm_86 -c cuda/aggregate.cu -o cuda/aggregate.o

# Then compile C++ files and link everything
nvcc -std=c++17 -arch=sm_86 -o sql \
  sql_parser.cpp \
  headers/column.cpp \
  headers/table.cpp \
  headers/duckdb_manager.cpp \
  headers/enums.cpp \
  headers/device_struct.cpp \
  cuda/aggregate.o \
  operators/aggregate.cpp \
  operators/join.cpp \
  -I$DUCKDB_INCLUDE -I. \
  -L$DUCKDB_LIB -lduckdb -lcudart \
  -Xlinker -rpath=$DUCKDB_LIB

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
else
    echo "Compilation successful!"
    # export LD_LIBRARY_PATH=$DUCKDB_LIB:$LD_LIBRARY_PATH
    # ./test_duckdb
    # echo "Execution completed with exit code $?"
fi
