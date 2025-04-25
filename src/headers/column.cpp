#include "column.h"
#include <stdexcept>

// Helper function to convert ColumnType to string
std::string columnTypeToString(ColumnType type)
{
    switch (type)
    {
    case ColumnType::STRING:
        return "STRING";
    case ColumnType::DOUBLE:
        return "DOUBLE";
    case ColumnType::DATE:
        return "DATE";
    default:
        return "UNKNOWN";
    }
}

// Helper function to convert string type to ColumnType
ColumnType stringToColumnType(const std::string &type_str)
{
    if (type_str == "VARCHAR")
        return ColumnType::STRING;
    if (type_str == "DOUBLE")
        return ColumnType::DOUBLE;
    if (type_str == "TIMESTAMP")
        return ColumnType::DATE;
    return ColumnType::UNKNOWN;
}

ColumnMetadata::ColumnMetadata(const std::string &name, ColumnType type, const std::string &duckdb_type,
                               bool is_primary_key, size_t index, size_t byte_offset, size_t element_size)
    : name(name), type(type), duckdb_type(duckdb_type), is_primary_key(is_primary_key),
      index(index), byte_offset(byte_offset), element_size(element_size) {}

ColumnBatch::ColumnBatch(ColumnType type, size_t expected_rows)
    : type(type), num_rows(0), on_gpu(false), gpu_data_ptr(nullptr)
{
    // Pre-allocate memory
    if (type == ColumnType::DOUBLE)
    {
        double_data.reserve(expected_rows);
    }
    else if (type == ColumnType::STRING)
    {
        string_data.reserve(expected_rows);
    }
    else if (type == ColumnType::DATE)
    {
        date_data.reserve(expected_rows);
    }
}

ColumnBatch::~ColumnBatch()
{
    // Release GPU memory if needed
    freeGpuMemory();
}

// Add data to the batch
void ColumnBatch::addDouble(double value)
{
    if (type == ColumnType::DOUBLE)
    {
        double_data.push_back(value);
        num_rows++;
    }
    else
    {
        throw std::runtime_error("Type mismatch: Cannot add double to " + columnTypeToString(type) + " column");
    }
}

void ColumnBatch::addString(const std::string &value)
{
    if (type == ColumnType::STRING)
    {
        string_data.push_back(value);
        num_rows++;
    }
    else
    {
        throw std::runtime_error("Type mismatch: Cannot add string to " + columnTypeToString(type) + " column");
    }
}

void ColumnBatch::addDate(const std::chrono::system_clock::time_point &value)
{
    if (type == ColumnType::DATE)
    {
        date_data.push_back(value);
        num_rows++;
    }
    else
    {
        throw std::runtime_error("Type mismatch: Cannot add date to " + columnTypeToString(type) + " column");
    }
}

// Get data
double ColumnBatch::getDouble(size_t row_idx) const
{
    if (type != ColumnType::DOUBLE || row_idx >= num_rows)
    {
        throw std::out_of_range("Invalid access to double data");
    }
    return double_data[row_idx];
}

const std::string &ColumnBatch::getString(size_t row_idx) const
{
    if (type != ColumnType::STRING || row_idx >= num_rows)
    {
        throw std::out_of_range("Invalid access to string data");
    }
    return string_data[row_idx];
}

std::chrono::system_clock::time_point ColumnBatch::getDate(size_t row_idx) const
{
    if (type != ColumnType::DATE || row_idx >= num_rows)
    {
        throw std::out_of_range("Invalid access to date data");
    }
    return date_data[row_idx];
}

// GPU operations (stubs to be implemented with actual CUDA code)
bool ColumnBatch::transferToGPU()
{
    // To be implemented with CUDA
    // This would allocate GPU memory and copy data to the GPU
    if (on_gpu)
        return true; // Already on GPU

    on_gpu = true; // Set this to true when implemented
    return on_gpu;
}

void ColumnBatch::freeGpuMemory()
{
    // To be implemented with CUDA
    // Free the GPU memory if allocated
    if (on_gpu && gpu_data_ptr)
    {
        // Call CUDA free
        on_gpu = false;
        gpu_data_ptr = nullptr;
    }
}

// Utilities
size_t ColumnBatch::size() const { return num_rows; }
ColumnType ColumnBatch::getType() const { return type; }
bool ColumnBatch::isOnGPU() const { return on_gpu; }