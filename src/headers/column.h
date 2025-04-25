#ifndef COLUMN_H
#define COLUMN_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

// Forward declaration of ColumnBatch
class ColumnBatch;

// Enum for column types
enum class ColumnType
{
    STRING, // VARCHAR/TEXT
    DOUBLE, // NUMERIC/FLOAT
    DATE,   // TIMESTAMP
    UNKNOWN
};

// Helper function to convert ColumnType to string
std::string columnTypeToString(ColumnType type);

// Helper function to convert string type to ColumnType
ColumnType stringToColumnType(const std::string &type_str);

// Struct for column metadata
struct ColumnMetadata
{
    std::string name;        // Column name
    ColumnType type;         // Column type (enum)
    std::string duckdb_type; // Original DuckDB type name
    bool is_primary_key;     // Flag to indicate if column is a primary key
    size_t index;            // Column index in table
    size_t byte_offset;      // Byte offset in the row structure
    size_t element_size;     // Size in bytes for the element (for fixed-size types)

    // Constructor
    ColumnMetadata(const std::string &name, ColumnType type, const std::string &duckdb_type,
                   bool is_primary_key, size_t index, size_t byte_offset = 0, size_t element_size = 0);
};

// Class for a batch of data from a column
class ColumnBatch
{
private:
    ColumnType type;
    size_t num_rows;

    // Data storage - using std::vector for flexibility
    std::vector<double> double_data;
    std::vector<std::string> string_data;
    std::vector<std::chrono::system_clock::time_point> date_data;

    // Flag to indicate if data is on GPU
    bool on_gpu;
    void *gpu_data_ptr; // GPU memory pointer (to be used with CUDA APIs)

public:
    ColumnBatch(ColumnType type, size_t expected_rows);
    ~ColumnBatch();

    // Add data to the batch
    void addDouble(double value);
    void addString(const std::string &value);
    void addDate(const std::chrono::system_clock::time_point &value);

    // Get data
    double getDouble(size_t row_idx) const;
    const std::string &getString(size_t row_idx) const;
    std::chrono::system_clock::time_point getDate(size_t row_idx) const;

    // GPU operations (stubs to be implemented with actual CUDA code)
    bool transferToGPU();
    void freeGpuMemory();

    // Utilities
    size_t size() const;
    ColumnType getType() const;
    bool isOnGPU() const;
};

#endif // COLUMN_H