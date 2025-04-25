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

bool Table::readNextBatch()
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
    std::cout << "Memory usage for this batch: " << formatMemorySize(getCurrentRSS()) << "\n";
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