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
    std::string save_file_path;
    std::streampos last_file_pos;                       // Last file position for resuming reads
    size_t number_of_rows;                                 // Number of rows read so far
    bool is_result_table = false;
    bool is_descending = false;
    std::string order_by_column; 
    size_t total_count;                                      // Current row in the file
    bool has_more_data;                                      // Flag to indicate if there's more data to read
    std::vector<std::shared_ptr<ColumnBatch>> current_batch; // Current batch of data
    std::vector<FilterCondition> filters;
    std::unordered_map<size_t, size_t> original_to_projected_map; 
    std::vector<size_t> projected_to_original_map; 
    void printRows(size_t start_row, size_t end_row, size_t max_string_length);

    
    
    public:
    Table(const std::string &name, const std::vector<ColumnMetadata> &columns,
        const std::string &file_path, size_t batch_size = int(1e7));
        bool passesFilters(const std::vector<std::string>& row_values) const;
        const std::vector<FilterCondition>& getFilters() const { return filters; }
        void addFilter(const FilterCondition& condition);
        void addFilters(const std::vector<FilterCondition>& conditions);
        void addProjectedColumns(const std::vector<std::size_t>& column_ids) ;
        void clearFilters();



    std::vector<DeviceStruct> transferBatchToGPU();
    std::vector<DeviceStruct> createSortStructs();
    void setIsResultTable(bool is_result_table) { this->is_result_table = is_result_table; }
    bool getIsResultTable() const { return is_result_table; }
    void setFilePath(const std::string &file_path) { this->file_path = file_path; }
    bool readNextBatch();
    void saveCurrentBatch();
    void printCurrentBatch(size_t max_rows = 10,size_t max_string_length = 30);
    std::vector<std::string> getProjectedColumnNames() const;
    std::vector<size_t> getProjectedColumnIndices() {return projected_to_original_map;};
    std::string getColumnName(size_t column_index) const;
    // Accessors
    const std::string &getName() const { return name; }
    const std::vector<ColumnMetadata> &getColumns() const { return columns; }
    size_t getColumnCount() const { return original_to_projected_map.size(); }
    bool hasMoreData() const { return has_more_data; }
    size_t getCurrentBatchSize() const;
    int64_t parseDate(const std::string& dateStr) const; 
    ColumnBatch *getColumnBatch(const std::string &column_name);
    ColumnBatch *getColumnBatch(size_t column_index);
    void createCSVHeaders();
    size_t getColumnIndexProjected(const std::string &column_name);
    size_t getColumnIndexOriginal(const std::string &column_name);
    std::vector<std::shared_ptr<ColumnBatch>> getCurrentBatch() const { return current_batch; }
    void resetFilePositionToStart();
    void setIsDescending(bool is_descending) { this->is_descending = is_descending; }
    bool getIsDescending() const { return is_descending; }
    size_t getTotalCount() const { return total_count; }

    ColumnType getColumnType(const std::string &column_name) ;
    void setCurrentBatch(const std::vector<std::shared_ptr<ColumnBatch>> &newBatch);
    void setSaveFilePath(const std::string &file_path);
    void addResultBatch(void **result_table_batches, size_t num_rows);
    friend class DuckDBManager;
};

#endif