#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
DUCKDB_INCLUDE="../duckdb/src/include"
DUCKDB_LIB="../duckdb/build/release/src"

# Compile CUDA files first with device code
nvcc -std=c++17 -arch=sm_75 -dc -o join.o \
  join.cu \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_75 -dc -o join_gpu.o \
  join_gpu.cu \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_75 -dc -o read_csv.o \
  read_csv.cu \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_75 -dc -o main.o \
  main.cu \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_75 -dc -o device_struct.o \
  ../headers/device_struct.cpp \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

nvcc -std=c++17 -arch=sm_75 -dc -o enums.o \
  ../headers/enums.cpp \
  -I. -I$DUCKDB_INCLUDE \
  -rdc=true

# Link everything together with device code linking enabled
nvcc -std=c++17 -arch=sm_75 -o join_example \
  main.o \
  join.o \
  join_gpu.o \
  read_csv.o \
  enums.o \
  device_struct.o \
  -I. -I$DUCKDB_INCLUDE \
  -L$DUCKDB_LIB -lduckdb \
  -lcudart \
  -Xlinker -rpath=$DUCKDB_LIB \
  -rdc=true

# Check if compilation was successful
if [ $? -ne 0 ]; then
  echo "Compilation failed!"
else
  echo "Compilation successful!"
  echo "Run the example with: ./join_example"
fi
