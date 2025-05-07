#include "column.h"
#include <stdexcept>
#include "cuda/aggregate.cuh"
#include "cuda/aggregate_helper.h"
#include "device_struct.h"
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
      index(index) {}

ColumnBatch::ColumnBatch(ColumnType type, size_t expected_rows)
    : type(type), num_rows(0), on_gpu(false)
{
    // Pre-allocate memory
    if (type == ColumnType::FLOAT)
    {
        float_data.reserve(expected_rows);
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
        float_data.push_back(value);
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

void ColumnBatch::addDate(const int64_t &value)
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
int64_t ColumnBatch::getDateAsInt64(size_t row_idx) const
{
    if (type != ColumnType::DATE || row_idx >= num_rows)
    {
        throw std::out_of_range("Invalid access to date data");
    }
    return date_data[row_idx];
}

// Get data
float ColumnBatch::getDouble(size_t row_idx) const
{
    if (type != ColumnType::FLOAT || row_idx >= num_rows)
    {
        throw std::out_of_range("Invalid access to float data");
    }
    return float_data[row_idx];
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
    return std::chrono::system_clock::time_point(std::chrono::nanoseconds(date_data[row_idx]));
}

// GPU operations (stubs to be implemented with actual CUDA code)
bool ColumnBatch::transferToGPU()
{
    // To be implemented with CUDA
    // This would allocate GPU memory and copy data to the GPU
    if (on_gpu)
        return true; // Already on GPU

    void *host_ptr = nullptr;
    switch (type)
    {
        case ColumnType::FLOAT:
            host_ptr = (void*)float_data.data();
            break;
        case ColumnType::STRING:
            host_ptr = (void*)string_data.data();
            break;
        case ColumnType::DATE:
            host_ptr = (void*)date_data.data();
            break;
        default:
            throw std::runtime_error("Unsupported column type for GPU transfer");
    }

    this->cpu_struct_ptr = DeviceStruct::createStruct(type, host_ptr, num_rows);
    
    on_gpu = true; // Set this to true when implemented
    return on_gpu;
}

void ColumnBatch::freeGpuMemory()
{
    // To be implemented with CUDA
    // Free the GPU memory if allocated
    if (cpu_struct_ptr != nullptr)
    {
        DeviceStruct::deleteStruct(*this->cpu_struct_ptr); // This is a unique_ptr, so it will automatically free the memory from the GPU
        delete this->cpu_struct_ptr;
        on_gpu = false;
    }
    
}

bool FilterCondition::evaluate(const FilterValue &row_value) const
{
    return std::visit([&](const auto &filter_val)
                      {
        // Handle the case where row_value and value types don't match
        if (!std::holds_alternative<std::decay_t<decltype(filter_val)>>(row_value)) {
            return false;
        }

        // Safe to get value since we've verified the types match
        auto row_val = std::get<std::decay_t<decltype(filter_val)>>(row_value);

        switch (op) {
            case ComparisonOperator::EQUALS:
                return row_val == filter_val;
            case ComparisonOperator::NOT_EQUALS:
                return row_val != filter_val;
            case ComparisonOperator::LESS_THAN:
                return row_val < filter_val;
            case ComparisonOperator::LESS_THAN_EQUALS:
                return row_val <= filter_val;
            case ComparisonOperator::GREATER_THAN:
                return row_val > filter_val;
            case ComparisonOperator::GREATER_THAN_EQUALS:
                return row_val >= filter_val;
            default:
                return false;
        } }, value);
}

// Utilities
size_t ColumnBatch::size() const { return num_rows; }
ColumnType ColumnBatch::getType() const { return type; }
bool ColumnBatch::isOnGPU() const { return on_gpu; }

std::string ColumnBatch::toString(size_t row_idx) const
{
    switch(type)
    {
        case ColumnType::FLOAT:
            return std::to_string(getDouble(row_idx));
        case ColumnType::DATE:
            return std::to_string(getDateAsInt64(row_idx));
        case ColumnType::STRING:
            return getString(row_idx);
        default:
            return "NULL";
    }
}

std::string operatorToString(ComparisonOperator op)
{
    switch (op)
    {
    case ComparisonOperator::EQUALS:
        return "=";
    case ComparisonOperator::NOT_EQUALS:
        return "!=";
    case ComparisonOperator::LESS_THAN:
        return "<";
    case ComparisonOperator::LESS_THAN_EQUALS:
        return "<=";
    case ComparisonOperator::GREATER_THAN:
        return ">";
    case ComparisonOperator::GREATER_THAN_EQUALS:
        return ">=";
    default:
        return "?";
    }
}
std::string FilterCondition::toString() const
{
    std::stringstream ss;
    ss << column_name << " " << operatorToString(op) << " ";

    std::visit([&](const auto &v)
               {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            ss << "'" << v << "'";
        } else if constexpr (std::is_same_v<T, int64_t>) {
            // Convert int64_t timestamp to formatted date string
            auto time_point = std::chrono::system_clock::time_point(std::chrono::nanoseconds(v));
            std::time_t time = std::chrono::system_clock::to_time_t(time_point);
            std::tm tm = *std::localtime(&time);
            char buffer[32];
            std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
            ss << "'" << buffer << "'";
        } else {
            ss << v;
        } }, value);

    return ss.str();
}
std::string ColumnBatch::computeAggregate(AggregateType agg_type)
{
    AggregateResult result;
    if (num_rows == 0)
    {
        std::cerr << "Warning: Attempting to aggregate empty column" << std::endl;
        return "NULL";
    }
    ValueType val_type;
    switch (type)
    {
    case ColumnType::FLOAT:
        val_type = TYPE_FLOAT;
        break;
    case ColumnType::DATE:
        val_type = TYPE_DATE;
        break;
    default:
        throw std::runtime_error("Unsupported column type for aggregation");
    }
    void *host_ptr = nullptr;
    if (type == ColumnType::FLOAT)
    {
        host_ptr = float_data.data();
    }
    else
    {
        host_ptr = date_data.data();
    }
    ::computeAggregate(host_ptr, num_rows, agg_type, val_type, result);
    std::string formatted = formatAggregateResult(result, type, agg_type);
    return formatted;
}