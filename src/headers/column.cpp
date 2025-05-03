#include "column.h"
#include <stdexcept>
#include <iomanip>
#include <sstream>
std::string columnTypeToString(ColumnType type)
{
    switch (type)
    {
    case ColumnType::STRING:
        return "STRING";
    case ColumnType::FLOAT:
        return "FLOAT";
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
    if (type_str == "FLOAT")
        return ColumnType::FLOAT;
    if (type_str == "TIMESTAMP")
        return ColumnType::DATE;
    return ColumnType::UNKNOWN;
}

ColumnMetadata::ColumnMetadata(const std::string &name, ColumnType type, const std::string &duckdb_type,
                               bool is_primary_key, size_t index)
    : name(name), type(type), duckdb_type(duckdb_type), is_primary_key(is_primary_key),
      index(index){}

ColumnBatch::ColumnBatch(ColumnType type, size_t expected_rows)
    : type(type), num_rows(0), on_gpu(false), gpu_data_ptr(nullptr)
{
    // Pre-allocate memory
    if (type == ColumnType::FLOAT)
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
void ColumnBatch::addDouble(float value)
{
    if (type == ColumnType::FLOAT)
    {
        double_data.push_back(value);
        num_rows++;
    }
    else
    {
        throw std::runtime_error("Type mismatch: Cannot add float to " + columnTypeToString(type) + " column");
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
float ColumnBatch::getDouble(size_t row_idx) const
{
    if (type != ColumnType::FLOAT || row_idx >= num_rows)
    {
        throw std::out_of_range("Invalid access to float data");
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

bool FilterCondition::evaluate(const FilterValue& row_value) const {
    return std::visit([&](const auto& filter_val) {
        // Handle the case where row_value and value types don't match
        if (!std::holds_alternative<std::decay_t<decltype(filter_val)>>(row_value)) {
            return false;
        }

        // Safe to get value since we've verified the types match
        auto row_val = std::get<std::decay_t<decltype(filter_val)>>(row_value);

        switch (op) {
            case FilterOperator::EQUALS:
                return row_val == filter_val;
            case FilterOperator::NOT_EQUALS:
                return row_val != filter_val;
            case FilterOperator::LESS_THAN:
                return row_val < filter_val;
            case FilterOperator::LESS_THAN_EQUALS:
                return row_val <= filter_val;
            case FilterOperator::GREATER_THAN:
                return row_val > filter_val;
            case FilterOperator::GREATER_THAN_EQUALS:
                return row_val >= filter_val;
            default:
                return false;
        }
    }, value);
}

// Utilities
size_t ColumnBatch::size() const { return num_rows; }
ColumnType ColumnBatch::getType() const { return type; }
bool ColumnBatch::isOnGPU() const { return on_gpu; }

std::string operatorToString(FilterOperator op) {
    switch (op) {
        case FilterOperator::EQUALS: return "=";
        case FilterOperator::NOT_EQUALS: return "!=";
        case FilterOperator::LESS_THAN: return "<";
        case FilterOperator::LESS_THAN_EQUALS: return "<=";
        case FilterOperator::GREATER_THAN: return ">";
        case FilterOperator::GREATER_THAN_EQUALS: return ">=";
        default: return "?";
    }
}
std::string FilterCondition::toString() const {
    std::stringstream ss;
    ss << column_name << " " << operatorToString(op) << " ";
    
    std::visit([&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            ss << "'" << v << "'";
        } else if constexpr (std::is_same_v<T, std::chrono::system_clock::time_point>) {
            std::time_t time = std::chrono::system_clock::to_time_t(v);
            std::tm tm = *std::localtime(&time);
            char buffer[32];
            std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
            ss << "'" << buffer << "'";
        } else {
            ss << v;
        }
    }, value);
    
    return ss.str();
}