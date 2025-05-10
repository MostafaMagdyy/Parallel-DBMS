export PATH=/usr/local/cuda/bin:$PATH
DUCKDB_INCLUDE="./duckdb/src/include"
DUCKDB_LIB="./duckdb/build/release/src"

# Compile CUDA files first
nvcc -std=c++17 -arch=sm_75 -dc -o cuda/aggregate.o \
  cuda/aggregate.cu \
  -I. -I$DUCKDB_INCLUDE
  
nvcc -std=c++17 -arch=sm_86 -dc -o RadixSort/scan.o \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_75 -dc -o RadixSort/scan.o \
  RadixSort/scan.cu \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_86 -dc -o RadixSort/sort.o \
  RadixSort/sort.cu \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_75 -dc -o Join/join_gpu.o \
  Join/join_gpu.cu \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_75 -dc -o Join/join.o \
  Join/join.cu \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

# Then compile C++ files and link everything
nvcc -std=c++17 -arch=sm_75 -o sql \
  RadixSort/scan.o \
  RadixSort/sort.o \
  cuda/aggregate.o \
  Join/join_gpu.o \
  Join/join.o \
  RadixSort/read_csv.cu \
  headers/column.cpp \
  headers/table.cpp \
  headers/duckdb_manager.cpp \
  headers/enums.cpp \
  headers/device_struct.cpp \
  operators/cpu_sort.cpp \
  operators/aggregate.cpp \
  operators/join.cpp \
  sql_parser.cpp \
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
