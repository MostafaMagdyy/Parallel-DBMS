#include "table.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
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

Table::Table(const std::string &name, const std::vector<ColumnMetadata> &columns,
             const std::string &file_path, size_t batch_size)
    : name(name), columns(columns), file_path(file_path),
      batch_size(batch_size), current_row(0), has_more_data(true), last_file_pos(0)
{

    // Initialize column map for quick lookups
    for (size_t i = 0; i < columns.size(); i++)
    {
        column_map[columns[i].name] = i;
    }
}
std::chrono::system_clock::time_point Table::parseDate(const std::string &dateStr) const
{
    // Handle empty or null values
    if (dateStr.empty() || dateStr == "NULL")
    {
        throw std::runtime_error("Invalid date format: " + dateStr +
                                 " (expected format: yyyy-MM-dd or yyyy-MM-dd HH:mm:ss)");
    }

    std::tm tm = {};
    if (dateStr.length() >= 10)
    { // yyyy-MM-dd
        if (sscanf(dateStr.c_str(), "%d-%d-%d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday) == 3)
        {
            tm.tm_year -= 1900; // Adjust year (tm_year is years since 1900)
            tm.tm_mon -= 1;     // Adjust month (tm_mon is 0-11)

            if (dateStr.length() >= 19)
            { // yyyy-MM-dd HH:mm:ss
                sscanf(dateStr.c_str() + 11, "%d:%d:%d", &tm.tm_hour, &tm.tm_min, &tm.tm_sec);
            }
            std::time_t time = std::mktime(&tm);
            return std::chrono::system_clock::from_time_t(time);
        }
    }

    // If parsing fails, throw an exception
    throw std::runtime_error("Invalid date format: " + dateStr +
                             " (expected format: yyyy-MM-dd or yyyy-MM-dd HH:mm:ss)");
}

bool Table::readNextBatch()
{
    if (!has_more_data)
    {
        std::cerr << "No more data to read from file: " << file_path << std::endl;
        return false;
    }
    try
    {

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

        std::string line;
        size_t rows_read = 0;
        std::cout << "Reading batch of " << batch_size << " rows" << std::endl;
        while (rows_read < batch_size && std::getline(file, line))
        {
            if (line.empty())
            {
                continue;
            }
            std::vector<std::string> fields;
            std::stringstream ss(line);
            std::string field;

            // This needs to be optimized
            // no need for this getline again
            // maybe we pass the whole line to pass filters
            // Split the line into fields on the go while checking filters
            // There is no way to do this now
            while (std::getline(ss, field, ','))
            {
                fields.push_back(field);
            }
            if (!passesFilters(fields))
            {
                continue;
            }
            

            // Process each column
            for (size_t col_idx = 0; col_idx < columns.size() && col_idx < fields.size(); col_idx++)
            {
                const std::string &field = fields[col_idx];

                switch (columns[col_idx].type)
                {
                case ColumnType::DOUBLE:
                    try
                    {
                        current_batch[col_idx]->addDouble(std::stod(field));
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "[ERROR] Line "
                                  << ", Column '" << columns[col_idx].name
                                  << "': Cannot convert '" << field
                                  << "' to double: " << e.what() << std::endl;
                    }
                    break;

                case ColumnType::STRING:
                    current_batch[col_idx]->addString(field);
                    break;

                case ColumnType::DATE:
                    try
                    {
                        current_batch[col_idx]->addDate(parseDate(field));
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "[ERROR] Line "
                                  << ", Column '" << columns[col_idx].name
                                  << "': Cannot parse date '" << field
                                  << "': " << e.what() << std::endl;
                        current_batch[col_idx]->addDate(std::chrono::system_clock::now());
                    }
                    break;

                default:
                    break;
                }
            }

            rows_read++;
        }
        last_file_pos = file.tellg();
        current_row += rows_read;
        has_more_data = !file.eof();
        size_t memory_after = getCurrentRSS();
        size_t memory_used = memory_after - memory_before;
        auto total_end_time = std::chrono::high_resolution_clock::now();

        // Debug output
        // This is the rows returned from this batch (ideally should be batch size)
        // Should add another var to track total rows read(during to applying filters)
        std::cout << "Read " << rows_read << " rows from file" << std::endl;
        std::cout << "Has more data: " << has_more_data << std::endl;
        std::chrono::duration<double> total_duration = total_end_time - total_start_time;
        std::cout << "Batch read time: " << total_duration.count() << " seconds" << std::endl;
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

void Table::printCurrentBatch(size_t max_rows, size_t max_string_length) const
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
}

// Helper method to print a range of rows
void Table::printRows(size_t start_row, size_t end_row, size_t max_string_length) const
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

size_t Table::getCurrentBatchSize() const
{
    return current_batch.empty() ? 0 : current_batch[0]->size();
}

// Column access by name
ColumnBatch *Table::getColumnBatch(const std::string &column_name)
{
    auto it = column_map.find(column_name);
    if (it == column_map.end())
    {
        return nullptr;
    }
    return current_batch[it->second].get();
}

// Column access by index
ColumnBatch *Table::getColumnBatch(size_t column_index)
{
    if (column_index >= current_batch.size())
    {
        return nullptr;
    }
    return current_batch[column_index].get();
}

// GPU operations
bool Table::transferBatchToGPU()
{
    bool success = true;
    for (auto &col_batch : current_batch)
    {
        success &= col_batch->transferToGPU();
    }
    return success;
}
bool Table::passesFilters(const std::vector<std::string> &row_values) const
{
    if (filters.empty())
    {
        return true;
    }

    for (const auto &filter : filters)
    {
        auto it = column_map.find(filter.getColumnName());
        if (it == column_map.end() || it->second >= row_values.size())
        {
            continue;
        }
        size_t col_idx = it->second;
        const std::string &field = row_values[col_idx];
        FilterCondition::FilterValue row_value;
        try
        {
            switch (columns[col_idx].type)
            {
            case ColumnType::DOUBLE:
                row_value = std::stod(field);
                break;

            case ColumnType::STRING:
                row_value = field;
                break;

            case ColumnType::DATE:
                row_value = parseDate(field);
                break;

            default:
                continue; // Skip
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] "
                      << ", Column '" << columns[col_idx].name
                      << "': Cannot convert '" << field
                      << "' to appropriate type: " << e.what() << std::endl;
            continue; // Skip this filter check on error
        }
        if (!filter.evaluate(row_value))
            return false;
    }

    // All filters passed
    return true;
}

void Table::addFilter(const FilterCondition &condition)
{
    filters.push_back(condition);
}

void Table::addFilters(const std::vector<FilterCondition> &conditions)
{
    filters.insert(filters.end(), conditions.begin(), conditions.end());
}

void Table::clearFilters()
{
    filters.clear();
}