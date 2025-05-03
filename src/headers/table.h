#ifndef TABLE_H
#define TABLE_H

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "column.h" 

class DuckDBManager;

class Table
{
private:
    std::string name;
    std::vector<ColumnMetadata> columns;
    std::unordered_map<std::string, size_t> column_map; // Maps column names to their indices
    size_t batch_size;                                  // Number of rows to read in one batch
    std::string file_path;                              // Path to the CSV file
    std::streampos last_file_pos;                       // Last file position for resuming reads

    // Current batch information
    size_t current_row;                                      // Current row in the file
    bool has_more_data;                                      // Flag to indicate if there's more data to read
    std::vector<std::unique_ptr<ColumnBatch>> current_batch; // Current batch of data
    std::vector<FilterCondition> filters;
    std::unordered_map<size_t, size_t> projected_columns_map; 
    void printRows(size_t start_row, size_t end_row, size_t max_string_length) const;
    
    
    public:
    Table(const std::string &name, const std::vector<ColumnMetadata> &columns,
        const std::string &file_path, size_t batch_size = 10000);
        bool passesFilters(const std::vector<std::string>& row_values) const;
        const std::vector<FilterCondition>& getFilters() const { return filters; }
        void addFilter(const FilterCondition& condition);
        void addFilters(const std::vector<FilterCondition>& conditions);
        void addProjectedColumns(const std::vector<std::size_t>& column_ids) ;
        void clearFilters();


    bool readNextBatch();
    void printCurrentBatch(size_t max_rows = 10,size_t max_string_length = 30) const;
    std::vector<std::string> getProjectedColumnNames() const;
    // Accessors
    const std::string &getName() const { return name; }
    const std::vector<ColumnMetadata> &getColumns() const { return columns; }
    size_t getColumnCount() const { return columns.size(); }
    bool hasMoreData() const { return has_more_data; }
    size_t getCurrentBatchSize() const;
    std::chrono::system_clock::time_point parseDate(const std::string& dateStr) const;
    ColumnBatch *getColumnBatch(const std::string &column_name);
    ColumnBatch *getColumnBatch(size_t column_index);
    bool transferBatchToGPU();
    friend class DuckDBManager;
};

#endif