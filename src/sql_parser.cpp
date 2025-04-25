#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <duckdb.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_join.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <chrono>
#include <sys/resource.h>

size_t getCurrentRSS()
{
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return (size_t)(rusage.ru_maxrss * 1024L); // Convert from KB to bytes
}

// Add this function to format memory size nicely
std::string formatMemorySize(size_t size_bytes)
{
    const char *units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(size_bytes);

    while (size >= 1024.0 && unit_index < 3)
    {
        size /= 1024.0;
        unit_index++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return ss.str();
}
using namespace duckdb;
namespace fs = std::filesystem;
// Add after the namespace declarations at the top
enum class ColumnType
{
    STRING, // VARCHAR/TEXT
    DOUBLE, // NUMERIC/FLOAT
    DATE,   // TIMESTAMP
    UNKNOWN
};

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
// Replace the existing ColumnMetadata struct
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
                   bool is_primary_key, size_t index, size_t byte_offset = 0, size_t element_size = 0)
        : name(name), type(type), duckdb_type(duckdb_type), is_primary_key(is_primary_key),
          index(index), byte_offset(byte_offset), element_size(element_size) {}
};
// Add after the ColumnMetadata struct

// Represents a batch of data from a column (columnar format)
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
    ColumnBatch(ColumnType type, size_t expected_rows)
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

    ~ColumnBatch()
    {
        // Release GPU memory if needed
        freeGpuMemory();
    }

    // Add data to the batch
    void addDouble(double value)
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

    void addString(const std::string &value)
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

    void addDate(const std::chrono::system_clock::time_point &value)
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
    double getDouble(size_t row_idx) const
    {
        if (type != ColumnType::DOUBLE || row_idx >= num_rows)
        {
            throw std::out_of_range("Invalid access to double data");
        }
        return double_data[row_idx];
    }

    const std::string &getString(size_t row_idx) const
    {
        if (type != ColumnType::STRING || row_idx >= num_rows)
        {
            throw std::out_of_range("Invalid access to string data");
        }
        return string_data[row_idx];
    }

    std::chrono::system_clock::time_point getDate(size_t row_idx) const
    {
        if (type != ColumnType::DATE || row_idx >= num_rows)
        {
            throw std::out_of_range("Invalid access to date data");
        }
        return date_data[row_idx];
    }

    // GPU operations (stubs to be implemented with actual CUDA code)
    bool transferToGPU()
    {
        // To be implemented with CUDA
        // This would allocate GPU memory and copy data to the GPU
        if (on_gpu)
            return true; // Already on GPU

        on_gpu = true; // Set this to true when implemented
        return on_gpu;
    }

    void freeGpuMemory()
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
    size_t size() const { return num_rows; }
    ColumnType getType() const { return type; }
    bool isOnGPU() const { return on_gpu; }
};

// Table structure that holds metadata and data
class Table
{
private:
    std::string name;
    std::vector<ColumnMetadata> columns;
    std::unordered_map<std::string, size_t> column_map; // Maps column names to indices
    size_t batch_size;                                  // Number of rows to read in one batch
    std::string file_path;                              // Path to the CSV file
    std::streampos last_file_pos;                       // Last file position for resuming reads

    // Current batch information
    size_t current_row;                                      // Current row in the file
    bool has_more_data;                                      // Flag to indicate if there's more data to read
    std::vector<std::unique_ptr<ColumnBatch>> current_batch; // Current batch of data

public:
    Table(const std::string &name, const std::vector<ColumnMetadata> &columns,
          const std::string &file_path, size_t batch_size = 10000)
        : name(name), columns(columns), file_path(file_path),
          batch_size(batch_size), current_row(0), has_more_data(true), last_file_pos(0)
    {

        // Initialize column map for quick lookups
        for (size_t i = 0; i < columns.size(); i++)
        {
            column_map[columns[i].name] = i;
        }
    }

    bool readNextBatch()
    {
        if (!has_more_data)
        {
            std::cerr << "No more data to read from file: " << file_path << std::endl;
            return false;
        }
        try
        {
            // Profile memory before loading

            // Reset current batch
            current_batch.clear();
            size_t memory_before = getCurrentRSS();
            auto total_start_time = std::chrono::high_resolution_clock::now();
            std::cout << "Memory usage before batch loading: " << formatMemorySize(memory_before) << std::endl;
            // Create column batches
            for (const auto &col : columns)
            {
                current_batch.push_back(std::make_unique<ColumnBatch>(col.type, batch_size));
            }

            // Open the file
            std::ifstream file(file_path);
            if (!file.is_open())
            {
                has_more_data = false;
                std::cerr << "[ERROR] Failed to open file: " << file_path << std::endl;
                return false;
            }

            // Skip header if it's the first batch
            if (current_row == 0)
            {
                std::string header;
                std::getline(file, header);
                last_file_pos = file.tellg();
                std::cout << "Skipped header row for table: " << name << std::endl;
            }
            else
            {
                std::cout << "Seeking to saved file position for row " << current_row << std::endl;
                file.seekg(last_file_pos);
                if (!file.good())
                {
                    has_more_data = false;
                    std::cerr << "[ERROR] Failed to seek to saved position in file: " << file_path << std::endl;
                    return false;
                }
            }

            // Read batch_size rows or until EOF
            std::string line;
            size_t rows_read = 0;
            size_t line_number = current_row + 1; // For error reporting

            std::cout << "Starting to read batch of up to " << batch_size << " rows" << std::endl;

            while (rows_read < batch_size && std::getline(file, line))
            {
                // Parse the CSV line
                std::stringstream ss(line);
                std::string field;
                size_t col_idx = 0;

                try
                {
                    while (std::getline(ss, field, ',') && col_idx < columns.size())
                    {
                        // Process the field based on column type
                        try
                        {
                            switch (columns[col_idx].type)
                            {
                            case ColumnType::DOUBLE:
                                try
                                {
                                    double val = std::stod(field);
                                    current_batch[col_idx]->addDouble(val);
                                }
                                catch (std::exception &e)
                                {
                                    // Handle conversion error

                                    std::cerr << "[ERROR] Line " << line_number
                                              << ", Column '" << columns[col_idx].name
                                              << "': Cannot convert '" << field
                                              << "' to double: " << e.what() << std::endl;
                                }
                                break;

                            case ColumnType::STRING:
                                current_batch[col_idx]->addString(field);
                                break;

                            case ColumnType::DATE:
                                // Simple date parsing logic - would need more robust parsing in practice
                                try
                                {
                                    // Just store current time as placeholder
                                    current_batch[col_idx]->addDate(std::chrono::system_clock::now());
                                }
                                catch (std::exception &e)
                                {
                                    std::cerr << "[ERROR] Line " << line_number
                                              << ", Column '" << columns[col_idx].name
                                              << "': Cannot parse date '" << field
                                              << "': " << e.what() << std::endl;
                                }
                                break;

                            default:
                                std::cerr << "[ERROR] Line " << line_number
                                          << ", Column '" << columns[col_idx].name
                                          << "': Unknown column type" << std::endl;

                                break;
                            }
                        }
                        catch (std::exception &e)
                        {
                            std::cerr << "[ERROR] Line " << line_number
                                      << ", Column '" << columns[col_idx].name
                                      << "': Exception during processing: " << e.what() << std::endl;
                        }

                        col_idx++;
                    }
                }
                catch (std::exception &e)
                {
                    std::cerr << "[ERROR] Exception processing line " << line_number
                              << ": " << e.what() << std::endl;
                }

                rows_read++;
                line_number++;
            }
            last_file_pos = file.tellg();

            // Update state
            current_row += rows_read;
            std::cout << "Read " << rows_read << " rows from file" << std::endl;
            has_more_data = (rows_read == batch_size);
            std::cout << "Has more data: " << has_more_data << std::endl;

            // Profile memory after loading
            size_t memory_after = getCurrentRSS();
            size_t memory_used = memory_after - memory_before;
            auto total_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> total_duration = total_end_time - total_start_time;
            std::cout << "Batch read time: " << total_duration.count() << " ms" << std::endl;
            std::cout << "Batch read complete: " << rows_read << " rows read" << std::endl;
            std::cout << "Memory usage after batch loading: " << formatMemorySize(memory_after) << std::endl;
            std::cout << "Memory increase for this batch: " << formatMemorySize(memory_used) << std::endl;

            return rows_read > 0;
        }
        catch (std::exception &e)
        {
            std::cerr << "[CRITICAL ERROR] Exception in readNextBatch: " << e.what() << std::endl;
            has_more_data = false;
            return false;
        }
    }

    void printCurrentBatch(size_t max_rows = 5, size_t max_string_length = 30) const
    {
        if (current_batch.empty())
        {
            std::cout << "No batch data available for table '" << name << "'." << std::endl;
            return;
        }

        size_t num_rows = current_batch[0]->size();
        size_t num_cols = columns.size();

        std::cout << "\n==== Table: " << name << " - Current Batch (" << num_rows
                  << " rows, " << num_cols << " columns) ====\n";

        // Print column headers
        std::cout << "Row# | ";
        for (const auto &col : columns)
        {
            std::string header = col.name;
            if (header.length() > max_string_length)
            {
                header = header.substr(0, max_string_length - 3) + "...";
            }
            std::cout << std::left << std::setw(max_string_length) << header << " | ";
        }
        std::cout << "\n";

        // Print separator
        std::string separator = "----+-";
        for (size_t i = 0; i < num_cols; i++)
        {
            separator += std::string(max_string_length, '-') + "-+-";
        }
        std::cout << separator << "\n";

        // Decide which rows to print
        if (num_rows <= 2 * max_rows)
        {
            // If we have fewer rows than 2*max_rows, just print them all
            printRows(0, num_rows, max_string_length);
        }
        else
        {
            // Print first max_rows
            printRows(0, max_rows, max_string_length);

            // Print a divider indicating skipped rows
            std::cout << ".... | " << std::string(num_cols * (max_string_length + 3), '.') << "\n";
            std::cout << "Skipped " << (num_rows - 2 * max_rows) << " rows\n";
            std::cout << ".... | " << std::string(num_cols * (max_string_length + 3), '.') << "\n";

            // Print last max_rows
            printRows(num_rows - max_rows, num_rows, max_string_length);
        }

        std::cout << separator << "\n";
        std::cout << "Memory usage for this batch: " << formatMemorySize(getCurrentRSS()) << "\n";
    }

private:
    // Helper method to print a range of rows
    void printRows(size_t start_row, size_t end_row, size_t max_string_length) const
    {
        for (size_t row = start_row; row < end_row; row++)
        {
            std::cout << std::setw(4) << row << " | ";

            for (size_t col = 0; col < columns.size(); col++)
            {
                std::string value;

                try
                {
                    switch (columns[col].type)
                    {
                    case ColumnType::DOUBLE:
                    {
                        double val = current_batch[col]->getDouble(row);
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(4) << val;
                        value = ss.str();
                        break;
                    }

                    case ColumnType::STRING:
                    {
                        value = current_batch[col]->getString(row);
                        if (value.length() > max_string_length)
                        {
                            value = value.substr(0, max_string_length - 3) + "...";
                        }
                        break;
                    }

                    case ColumnType::DATE:
                    {
                        auto time_point = current_batch[col]->getDate(row);
                        std::time_t time = std::chrono::system_clock::to_time_t(time_point);
                        std::tm tm = *std::localtime(&time);
                        char buffer[32];
                        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
                        value = buffer;
                        break;
                    }

                    default:
                        value = "[unknown type]";
                        break;
                    }
                }
                catch (const std::exception &e)
                {
                    value = "[error: " + std::string(e.what()) + "]";
                }

                std::cout << std::left << std::setw(max_string_length) << value << " | ";
            }
            std::cout << "\n";
        }
    }

    // Accessors
    const std::string &getName() const { return name; }
    const std::vector<ColumnMetadata> &getColumns() const { return columns; }
    size_t getColumnCount() const { return columns.size(); }
    bool hasMoreData() const { return has_more_data; }
    size_t getCurrentBatchSize() const
    {
        return current_batch.empty() ? 0 : current_batch[0]->size();
    }

    // Column access by name
    ColumnBatch *getColumnBatch(const std::string &column_name)
    {
        auto it = column_map.find(column_name);
        if (it == column_map.end())
        {
            return nullptr;
        }
        return current_batch[it->second].get();
    }

    // Column access by index
    ColumnBatch *getColumnBatch(size_t column_index)
    {
        if (column_index >= current_batch.size())
        {
            return nullptr;
        }
        return current_batch[column_index].get();
    }

    // GPU operations
    bool transferBatchToGPU()
    {
        bool success = true;
        for (auto &col_batch : current_batch)
        {
            success &= col_batch->transferToGPU();
        }
        return success;
    }
};

static std::vector<ColumnMetadata> parseCSVHeader(const std::string &csv_file)
{
    std::ifstream file(csv_file);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open CSV file: " + csv_file);
    }

    std::string header_line;
    std::getline(file, header_line);
    std::stringstream ss(header_line);
    std::string token;
    std::vector<ColumnMetadata> columns;
    size_t index = 0;
    size_t byte_offset = 0; // Track byte offset for columnar layout

    while (std::getline(ss, token, ','))
    {
        size_t type_start = token.find('(');
        if (type_start == std::string::npos)
        {
            throw std::runtime_error("Invalid header format in CSV (missing type): " + token);
        }

        std::string col_name = token.substr(0, type_start);
        // Remove leading/trailing whitespace from column name
        col_name.erase(0, col_name.find_first_not_of(" \t"));
        col_name.erase(col_name.find_last_not_of(" \t") + 1);

        // Process all parenthesized parts
        std::string duckdb_type;
        bool is_primary_key = false;
        ColumnType col_type = ColumnType::UNKNOWN;
        size_t element_size = 0;

        size_t pos = 0;
        while ((pos = token.find('(', pos)) != std::string::npos)
        {
            size_t end_pos = token.find(')', pos);
            if (end_pos == std::string::npos)
            {
                throw std::runtime_error("Invalid header format in CSV (unclosed parenthesis): " + token);
            }

            // Extract content inside parentheses
            std::string part = token.substr(pos + 1, end_pos - pos - 1);
            part.erase(0, part.find_first_not_of(" \t"));
            part.erase(part.find_last_not_of(" \t") + 1);

            // Determine what this part represents
            if (part == "T")
            {
                duckdb_type = "VARCHAR";
                col_type = ColumnType::STRING;
                element_size = 0; // Variable size
            }
            else if (part == "N")
            {
                duckdb_type = "DOUBLE";
                col_type = ColumnType::DOUBLE;
                element_size = sizeof(double);
            }
            else if (part == "D")
            {
                duckdb_type = "TIMESTAMP";
                col_type = ColumnType::DATE;
                element_size = sizeof(int64_t); // Store as epoch timestamp
            }
            else if (part == "P")
            {
                is_primary_key = true;
            }
            else
            {
                throw std::runtime_error("Unsupported type or constraint in header: " + part);
            }

            pos = end_pos + 1;
        }

        // Make sure we have a type
        if (col_type == ColumnType::UNKNOWN)
        {
            throw std::runtime_error("No valid type specified for column: " + col_name);
        }

        columns.push_back(ColumnMetadata(col_name, col_type, duckdb_type, is_primary_key,
                                         index, byte_offset, element_size));

        // Update byte offset and index for next column
        byte_offset += (element_size > 0) ? element_size : sizeof(void *); // For variable-sized types, store pointers
        index++;
    }

    file.close();
    return columns;
}
// Function to traverse get table names from logical operators
void getTableNames(LogicalOperator *op, std::vector<std::string> &table_names)
{
    if (!op)
        return;

    if (op->type == LogicalOperatorType::LOGICAL_GET)
    {
        auto get = reinterpret_cast<LogicalGet *>(op);
        if (auto table = get->GetTable())
        {
            table_names.push_back(table->name);
        }
        else if (!get->function.name.empty())
        {
            table_names.push_back(get->function.name);
        }
    }

    // Recursively check children
    for (auto &child : op->children)
    {
        getTableNames(child.get(), table_names);
    }
}
// Function to traverse the logical operator tree
void traverseLogicalOperator(LogicalOperator *op, int depth = 0)
{
    if (!op)
        return;

    // Print indentation based on depth
    std::string indent(depth * 2, ' ');

    // Print information about the current operator
    std::cout << indent << "Operator Type: " << op->GetName() << std::endl;

    // Print operator-specific information
    switch (op->type)
    {
    case LogicalOperatorType::LOGICAL_PROJECTION:
    {
        auto projection = reinterpret_cast<LogicalProjection *>(op);
        std::cout << indent << "Expressions: ";
        for (size_t i = 0; i < projection->expressions.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << projection->expressions[i]->ToString();
        }
        std::cout << std::endl;
        break;
    }
    case LogicalOperatorType::LOGICAL_FILTER:
    {
        auto filter = reinterpret_cast<LogicalFilter *>(op);
        std::cout << indent << "Filter Expressions: ";
        for (size_t i = 0; i < filter->expressions.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << filter->expressions[i]->ToString();
        }
        std::cout << std::endl;
        break;
    }
    case LogicalOperatorType::LOGICAL_GET:
    {
        auto get = reinterpret_cast<LogicalGet *>(op);
        // Use GetTable() to get table information if available
        if (auto table = get->GetTable())
        {
            std::cout << indent << "Table: " << table->name << std::endl;
        }
        else
        {
            std::cout << indent << "Table: " << "(Function Scan)" << std::endl;
        }
        // Print returned column names
        if (!get->table_filters.filters.empty())
        {
            std::cout << indent << "Pushed-down Filters:" << std::endl;
            for (auto &kv : get->table_filters.filters)
            {
                auto &column_index = kv.first;
                auto &filter = kv.second;
                // Get the column name if available
                string column_name;
                if (column_index < get->names.size())
                {
                    column_name = get->names[column_index];
                }
                else
                {
                    column_name = "col_" + std::to_string(column_index);
                }
                // Use the filter's ToString method with the column name
                std::cout << indent << "  " << filter->ToString(column_name) << std::endl;
            }
        }
        std::cout << std::endl;
        break;
    }
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    {
        auto join = reinterpret_cast<LogicalComparisonJoin *>(op);
        std::cout << indent << "Join Type: " << JoinTypeToString(join->join_type) << std::endl;

        // Print join conditions if available
        if (!join->conditions.empty())
        {
            std::cout << indent << "Join Conditions: ";
            for (size_t i = 0; i < join->conditions.size(); i++)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << join->conditions[i].left->ToString() << " "
                          << ExpressionTypeToOperator(join->conditions[i].comparison) << " "
                          << join->conditions[i].right->ToString();
            }
            std::cout << std::endl;
        }
        else if (!join->expressions.empty())
        {
            // Some joins might still use expressions for conditions
            std::cout << indent << "Join Expressions: ";
            for (size_t i = 0; i < join->expressions.size(); i++)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << join->expressions[i]->ToString();
            }
            std::cout << std::endl;
        }

        std::unordered_set<idx_t> left_bindings;
        std::unordered_set<idx_t> right_bindings;

        if (!join->children.empty() && join->children.size() >= 2)
        {

            // Additionally, get the actual table names
            std::vector<std::string> left_tables;
            std::vector<std::string> right_tables;

            getTableNames(join->children[0].get(), left_tables);
            getTableNames(join->children[1].get(), right_tables);

            std::cout << indent << "Left Tables: ";
            for (size_t i = 0; i < left_tables.size(); i++)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << left_tables[i];
            }
            std::cout << std::endl;

            std::cout << indent << "Right Tables: ";
            for (size_t i = 0; i < right_tables.size(); i++)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << right_tables[i];
            }
            std::cout << std::endl;
        }

        // Print predicate if available (your existing code)
        if (join->predicate)
        {
            std::cout << indent << "Additional Predicate: " << join->predicate->ToString() << std::endl;
        }

        break;
    }
    case LogicalOperatorType::LOGICAL_ORDER_BY:
    {
        auto order = reinterpret_cast<LogicalOrder *>(op);
        std::cout << indent << "Order By: ";
        for (size_t i = 0; i < order->orders.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << order->orders[i].expression->ToString() << " "
                      << (order->orders[i].type == OrderType::ASCENDING ? "ASC" : "DESC");
        }
        std::cout << std::endl;
        break;
    }
    default:
        break;
    }

    std::cout << indent << "------------------------" << std::endl;

    // Recursively traverse child nodes
    for (auto &child : op->children)
    {
        traverseLogicalOperator(child.get(), depth + 1);
    }
}
// Class to manage DuckDB operations
class DuckDBManager
{
private:
    std::unique_ptr<DuckDB> db;
    std::unique_ptr<Connection> con;
    std::unordered_map<std::string, std::shared_ptr<Table>> tables;
    size_t default_batch_size;

    // Private constructor to prevent direct instantiation without initialization
    DuckDBManager() : db(nullptr), con(nullptr) {}

public:
    static DuckDBManager create()
    {
        DuckDBManager manager;
        manager.db = std::make_unique<DuckDB>(nullptr);
        manager.con = std::make_unique<Connection>(*manager.db);
        manager.con->Query("SET disabled_optimizers = 'statistics_propagation, filter_pushdown';");
        manager.default_batch_size = 1000000; // Default batch size
        return manager;
    }

    ~DuckDBManager()
    {
        con.reset();
        db.reset();
    }

    // Delete copy constructor and assignment operator to prevent copying
    DuckDBManager(const DuckDBManager &) = delete;
    DuckDBManager &operator=(const DuckDBManager &) = delete;

    // Move constructor and assignment operator for safe transfer
    DuckDBManager(DuckDBManager &&) = default;
    DuckDBManager &operator=(DuckDBManager &&) = default;

    static std::vector<ColumnMetadata> parseCSVHeader(const std::string &csv_file)
    {
        std::ifstream file(csv_file);
        if (!file.is_open())
        {
            throw std::runtime_error("Unable to open CSV file: " + csv_file);
        }

        std::string header_line;
        std::getline(file, header_line);
        std::stringstream ss(header_line);
        std::string token;
        std::vector<ColumnMetadata> columns;
        size_t index = 0;
        size_t byte_offset = 0;

        while (std::getline(ss, token, ','))
        {
            size_t type_start = token.find('(');
            if (type_start == std::string::npos)
            {
                throw std::runtime_error("Invalid header format in CSV (missing type): " + token);
            }

            std::string col_name = token.substr(0, type_start);
            // Remove leading/trailing whitespace from column name
            col_name.erase(0, col_name.find_first_not_of(" \t"));
            col_name.erase(col_name.find_last_not_of(" \t") + 1);

            // Process all parenthesized parts
            std::string duckdb_type;
            bool is_primary_key = false;
            ColumnType col_type = ColumnType::UNKNOWN;
            size_t element_size = 0;

            size_t pos = 0;
            while ((pos = token.find('(', pos)) != std::string::npos)
            {
                size_t end_pos = token.find(')', pos);
                if (end_pos == std::string::npos)
                {
                    throw std::runtime_error("Invalid header format in CSV (unclosed parenthesis): " + token);
                }

                // Extract content inside parentheses
                std::string part = token.substr(pos + 1, end_pos - pos - 1);
                part.erase(0, part.find_first_not_of(" \t"));
                part.erase(part.find_last_not_of(" \t") + 1);

                // Determine what this part represents
                if (part == "T")
                {
                    duckdb_type = "VARCHAR";
                    col_type = ColumnType::STRING;
                    element_size = 0; // Variable size
                }
                else if (part == "N")
                {
                    duckdb_type = "DOUBLE";
                    col_type = ColumnType::DOUBLE;
                    element_size = sizeof(double);
                }
                else if (part == "D")
                {
                    duckdb_type = "TIMESTAMP";
                    col_type = ColumnType::DATE;
                    element_size = sizeof(int64_t); // Store as epoch timestamp
                }
                else if (part == "P")
                {
                    is_primary_key = true;
                }
                else
                {
                    throw std::runtime_error("Unsupported type or constraint in header: " + part);
                }

                pos = end_pos + 1;
            }

            // Make sure we have a type
            if (col_type == ColumnType::UNKNOWN)
            {
                throw std::runtime_error("No valid type specified for column: " + col_name);
            }

            // Use the correct constructor signature
            columns.push_back(ColumnMetadata(
                col_name,       // name
                col_type,       // type (enum)
                duckdb_type,    // duckdb_type
                is_primary_key, // is_primary_key
                index,          // index
                byte_offset,    // byte_offset
                element_size    // element_size
                ));

            // Update byte offset and index for next column
            byte_offset += (element_size > 0) ? element_size : sizeof(void *); // For variable-sized types, store pointers
            index++;
        }

        file.close();
        return columns;
    }

    // Static function to create table in DuckDB from CSV
    static void createTableFromCSV(DuckDB &db, Connection &con, const std::string &csv_file)
    {
        std::string table_name = fs::path(csv_file).stem().string();
        auto columns = parseCSVHeader(csv_file);

        // Build CREATE TABLE statement
        std::stringstream create_sql;
        create_sql << "CREATE TABLE " << table_name << " (";
        for (size_t i = 0; i < columns.size(); ++i)
        {
            if (i > 0)
                create_sql << ", ";
            create_sql << columns[i].name << " " << columns[i].duckdb_type;
            if (columns[i].is_primary_key)
            {
                create_sql << " PRIMARY KEY";
            }
        }
        create_sql << ");";

        // Execute CREATE TABLE
        con.Query(create_sql.str());
        std::cout << "Query is " << create_sql.str() << std::endl;
        std::cout << "Created table: " << table_name << " (schema only) from " << csv_file << std::endl;
    }

    std::shared_ptr<Table> getTable(const std::string &table_name)
    {
        auto it = tables.find(table_name);
        if (it != tables.end())
        {
            return it->second;
        }
        return nullptr;
    }
    std::unique_ptr<LogicalOperator> getQueryPlan(const std::string &query)
    {
        if (!con)
        {
            throw std::runtime_error("DuckDB connection not initialized.");
        }
        std::cout << "Generating query plan for: " << query << std::endl;
        return con->ExtractPlan(query);
    }

    // Add table management functions
    void setBatchSize(size_t batch_size)
    {
        default_batch_size = batch_size;
    }

    // Load tables from CSV
    void loadTableFromCSV(const std::string &csv_file, size_t batch_size = 0)
    {
        if (!db || !con)
        {
            throw std::runtime_error("DuckDB not initialized.");
        }

        // Use default batch size if not specified
        if (batch_size == 0)
        {
            batch_size = default_batch_size;
        }

        std::string table_name = fs::path(csv_file).stem().string();
        auto columns = parseCSVHeader(csv_file);

        // Create DuckDB table (for query planning)
        createTableFromCSV(*db, *con, csv_file);

        // Create our own table representation
        auto table = std::make_shared<Table>(table_name, columns, csv_file, batch_size);
        tables[table_name] = table;
    }

    // Initialize all tables from CSV files in a directory
    void initializeTablesFromCSVs(const std::string &csv_directory, size_t batch_size = 0)
    {
        if (!db || !con)
        {
            throw std::runtime_error("DuckDB not initialized.");
        }

        for (const auto &entry : fs::directory_iterator(csv_directory))
        {
            if (entry.path().extension() == ".csv")
            {
                loadTableFromCSV(entry.path().string(), batch_size);
            }
        }
    }
    bool readNextBatch(const std::string &table_name)
    {
        auto it = tables.find(table_name);
        if (it != tables.end())
        {
            bool res = it->second->readNextBatch();
            it->second->printCurrentBatch();
            return res;
        }
        return false;
    }
};

int main()
{
    try
    {
        DuckDBManager db_manager = DuckDBManager::create();

        // Initialize tables from CSV files in a directory (schema only)
        std::string csv_directory = "./csv_data";
        db_manager.initializeTablesFromCSVs(csv_directory);
        // db_manager.listTables();
        std::vector<std::string> test_queries = {
            //     // Simple select
            //     // "SELECT name FROM users WHERE age > 25 AND dept_id = 1",
            //     "SELECT e.name, e.salary, d.department_name " \
        //     "FROM employees e, departments d " \
        //     "WHERE e.salary > 50000 " \
        //     "AND d.department_name != 'HR' " \
        //     "AND e.department_id = d.id " \
        //     "ORDER BY e.salary DESC"
            "SELECT name, salary, hire_date "
            "FROM employees "
            "WHERE salary > 50000 "
            "ORDER BY salary DESC"};
        std::cout << "=========================================" << std::endl;
        std::cout << db_manager.readNextBatch("employees") << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << db_manager.readNextBatch("employees") << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << db_manager.readNextBatch("employees") << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << db_manager.readNextBatch("employees") << std::endl;
        // std::cout << "=========================================" << std::endl;
        // std::cout << db_manager.readNextBatch("employees") << std::endl;
        // std::cout << "=========================================" << std::endl;
        // std::cout << db_manager.readNextBatch("employees") << std::endl;
        // for (auto &query : test_queries)
        // {
        //     std::cout << "\n=========================================" << std::endl;
        //     std::cout << "Planning query: " << query << std::endl;
        //     std::cout << "=========================================\n"
        //               << std::endl;

        //     // Get the logical plan
        //     auto plan = db_manager.getQueryPlan(query);
        //     // Print the default tree visualization
        //     std::cout << "Default plan visualization:" << std::endl;
        //     plan->Print();
        //     std::cout << std::endl;
        //     // Use our custom traversal function
        //     std::cout << "Custom tree traversal:" << std::endl;
        //     traverseLogicalOperator(plan.get());
        //     std::cout << std::endl;
        // }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}