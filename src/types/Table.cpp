// Table structure that holds metadata and data
class Table {
    private:
        std::string name;
        std::vector<ColumnMetadata> columns;
        std::unordered_map<std::string, size_t> column_map;  // Maps column names to indices
        size_t batch_size;  // Number of rows to read in one batch
        std::string file_path;  // Path to the CSV file
        
        // Current batch information
        size_t current_row;  // Current row in the file
        bool has_more_data;  // Flag to indicate if there's more data to read
        std::vector<std::unique_ptr<ColumnBatch>> current_batch;  // Current batch of data
    
    public:
        Table(const std::string& name, const std::vector<ColumnMetadata>& columns, 
              const std::string& file_path, size_t batch_size = 10000)
            : name(name), columns(columns), file_path(file_path), 
              batch_size(batch_size), current_row(0), has_more_data(true) {
            
            // Initialize column map for quick lookups
            for (size_t i = 0; i < columns.size(); i++) {
                column_map[columns[i].name] = i;
            }
        }
        
        // Initialize by reading the first batch
        bool initialize() {
            return readNextBatch();
        }
        
        // Read the next batch of data from file
        bool readNextBatch() {
            // Reset current batch
            current_batch.clear();
            
            // Create column batches
            for (const auto& col : columns) {
                current_batch.push_back(std::make_unique<ColumnBatch>(col.type, batch_size));
            }
            
            // Open the file
            std::ifstream file(file_path);
            if (!file.is_open()) {
                has_more_data = false;
                return false;
            }
            
            // Skip header if it's the first batch
            if (current_row == 0) {
                std::string header;
                std::getline(file, header);
            } else {
                // Seek to the current position
                size_t pos_to_skip = 0;
                std::string line;
                for (size_t i = 0; i < current_row; i++) {
                    if (!std::getline(file, line)) {
                        has_more_data = false;
                        return false;
                    }
                }
            }
            
            // Read batch_size rows or until EOF
            std::string line;
            size_t rows_read = 0;
            
            while (rows_read < batch_size && std::getline(file, line)) {
                // Parse the CSV line
                std::stringstream ss(line);
                std::string field;
                size_t col_idx = 0;
                
                while (std::getline(ss, field, ',') && col_idx < columns.size()) {
                    // Process the field based on column type
                    switch (columns[col_idx].type) {
                        case ColumnType::DOUBLE:
                            try {
                                double val = std::stod(field);
                                current_batch[col_idx]->addDouble(val);
                            } catch (std::exception& e) {
                                // Handle conversion error
                                current_batch[col_idx]->addDouble(0.0);  // Default value
                            }
                            break;
                            
                        case ColumnType::STRING:
                            current_batch[col_idx]->addString(field);
                            break;
                            
                        case ColumnType::DATE:
                            // Simple date parsing logic - would need more robust parsing in practice
                            try {
                                // Just store current time as placeholder
                                current_batch[col_idx]->addDate(std::chrono::system_clock::now());
                            } catch (std::exception& e) {
                                // Handle conversion error
                                current_batch[col_idx]->addDate(std::chrono::system_clock::now());
                            }
                            break;
                            
                        default:
                            // Unknown type
                            break;
                    }
                    
                    col_idx++;
                }
                
                rows_read++;
            }
            
            // Update state
            current_row += rows_read;
            has_more_data = (rows_read == batch_size);  // More data if we read a full batch
            
            return rows_read > 0;
        }
        
        // Accessors
        const std::string& getName() const { return name; }
        const std::vector<ColumnMetadata>& getColumns() const { return columns; }
        size_t getColumnCount() const { return columns.size(); }
        bool hasMoreData() const { return has_more_data; }
        size_t getCurrentBatchSize() const {
            return current_batch.empty() ? 0 : current_batch[0]->size();
        }
        
        // Column access by name
        ColumnBatch* getColumnBatch(const std::string& column_name) {
            auto it = column_map.find(column_name);
            if (it == column_map.end()) {
                return nullptr;
            }
            return current_batch[it->second].get();
        }
        
        // Column access by index
        ColumnBatch* getColumnBatch(size_t column_index) {
            if (column_index >= current_batch.size()) {
                return nullptr;
            }
            return current_batch[column_index].get();
        }
        
        // GPU operations
        bool transferBatchToGPU() {
            bool success = true;
            for (auto& col_batch : current_batch) {
                success &= col_batch->transferToGPU();
            }
            return success;
        }
    };