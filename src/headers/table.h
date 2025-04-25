#ifndef TABLE_H
#define TABLE_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <unordered_map>
#include "column.h" // Include the column header

// Forward declaration of DuckDBManager
class DuckDBManager;

// Class for a table
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
          const std::string &file_path, size_t batch_size = 10000);

    bool readNextBatch();
    void printCurrentBatch(size_t max_rows = 5, size_t max_string_length = 30) const;

private:
    // Helper method to print a range of rows
    void printRows(size_t start_row, size_t end_row, size_t max_string_length) const;

public:
    // Accessors
    const std::string &getName() const { return name; }
    const std::vector<ColumnMetadata> &getColumns() const { return columns; }
    size_t getColumnCount() const { return columns.size(); }
    bool hasMoreData() const { return has_more_data; }
    size_t getCurrentBatchSize() const;

    // Column access by name
    ColumnBatch *getColumnBatch(const std::string &column_name);

    // Column access by index
    ColumnBatch *getColumnBatch(size_t column_index);

    // GPU operations
    bool transferBatchToGPU();
    friend class DuckDBManager;
};

#endif // TABLE_H