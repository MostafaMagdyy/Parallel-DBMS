
export PATH=/usr/local/cuda/bin:$PATH
DUCKDB_INCLUDE="./duckdb/src/include"
DUCKDB_LIB="./duckdb/build/release/src" 
nvcc -std=c++17 -o test_duckdb sql_parser.cpp -I$DUCKDB_INCLUDE -L$DUCKDB_LIB -lduckdb -Xlinker -rpath=$DUCKDB_LIB

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
export LD_LIBRARY_PATH=$DUCKDB_LIB:$LD_LIBRARY_PATH
./test_duckdb

echo "Execution completed with exit code $?"