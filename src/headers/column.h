#ifndef COLUMN_H
#define COLUMN_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <variant>
#include <functional>
#include "../cuda/aggregate.cuh"
class ColumnBatch;

enum class ColumnType
{
    STRING, // VARCHAR/TEXT
    FLOAT, // NUMERIC/FLOAT
    DATE,   // TIMESTAMP
    UNKNOWN
};
enum class FilterOperator {
    EQUALS,
    NOT_EQUALS,
    LESS_THAN,
    LESS_THAN_EQUALS,
    GREATER_THAN,
    GREATER_THAN_EQUALS
};

class FilterCondition {
public:
    // The value a filter can check against (supports all column types)
    using FilterValue = std::variant<float, std::string,int64_t>;
    
    FilterCondition(const std::string& column_name, FilterOperator op, FilterValue value)
        : column_name(column_name), op(op), value(std::move(value)) {}
    
    // Evaluate the condition against a value from a table row
    bool evaluate(const FilterValue& row_value) const;
    
    std::string toString() const;
    const std::string& getColumnName() const { return column_name; }

private:
    std::string column_name;  // Name of column to filter on
    FilterOperator op;        // Comparison operator
    FilterValue value;        // Value to compare against
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

    // Constructor
    ColumnMetadata(const std::string &name, ColumnType type, const std::string &duckdb_type,
                   bool is_primary_key, size_t index);
};

class ColumnBatch
{
private:
    ColumnType type;
    size_t num_rows;

    std::vector<float> float_data;
    std::vector<std::string> string_data;
    std::vector<int64_t> date_data;

    bool on_gpu;
    void *gpu_data_ptr; // GPU memory pointer (to be used with CUDA APIs)

public:
    ColumnBatch(ColumnType type, size_t expected_rows);
    ~ColumnBatch();

    // Add data to the batch
    void addDouble(float value);
    void addString(const std::string &value);
    void addDate(const int64_t &value);

    // Get data
    float getDouble(size_t row_idx) const;
    const std::string &getString(size_t row_idx) const;
    int64_t getDateAsInt64(size_t row_idx) const;
    std::chrono::system_clock::time_point getDate(size_t row_idx) const;

    // GPU operations (stubs to be implemented with actual CUDA code)
    bool transferToGPU();
    void freeGpuMemory();
    std::string computeAggregate(AggregateType agg_type);

    // Utilities
    size_t size() const;
    ColumnType getType() const;
    bool isOnGPU() const;
};

#endif