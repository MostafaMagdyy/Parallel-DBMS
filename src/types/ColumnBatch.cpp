#include <Column.cpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
class ColumnBatch {
    private:
        ColumnType type;
        size_t num_rows;
        
        // Data storage - using std::vector for flexibility
        std::vector<double> double_data;
        std::vector<std::string> string_data;
        std::vector<std::chrono::system_clock::time_point> date_data;
        
        // Flag to indicate if data is on GPU
        bool on_gpu;
        void* gpu_data_ptr;  // GPU memory pointer (to be used with CUDA APIs)
    
    public:
        ColumnBatch(ColumnType type, size_t expected_rows) 
            : type(type), num_rows(0), on_gpu(false), gpu_data_ptr(nullptr) {
            // Pre-allocate memory
            if (type == ColumnType::DOUBLE) {
                double_data.reserve(expected_rows);
            } else if (type == ColumnType::STRING) {
                string_data.reserve(expected_rows);
            } else if (type == ColumnType::DATE) {
                date_data.reserve(expected_rows);
            }
        }
        
        ~ColumnBatch() {
            // Release GPU memory if needed
            freeGpuMemory();
        }
        
        // Add data to the batch
        void addDouble(double value) {
            if (type == ColumnType::DOUBLE) {
                double_data.push_back(value);
                num_rows++;
            } else {
                throw std::runtime_error("Type mismatch: Cannot add double to " + columnTypeToString(type) + " column");
            }
        }
        
        void addString(const std::string& value) {
            if (type == ColumnType::STRING) {
                string_data.push_back(value);
                num_rows++;
            } else {
                throw std::runtime_error("Type mismatch: Cannot add string to " + columnTypeToString(type) + " column");
            }
        }
        
        void addDate(const std::chrono::system_clock::time_point& value) {
            if (type == ColumnType::DATE) {
                date_data.push_back(value);
                num_rows++;
            } else {
                throw std::runtime_error("Type mismatch: Cannot add date to " + columnTypeToString(type) + " column");
            }
        }
        
        // Get data
        double getDouble(size_t row_idx) const {
            if (type != ColumnType::DOUBLE || row_idx >= num_rows) {
                throw std::out_of_range("Invalid access to double data");
            }
            return double_data[row_idx];
        }
        
        const std::string& getString(size_t row_idx) const {
            if (type != ColumnType::STRING || row_idx >= num_rows) {
                throw std::out_of_range("Invalid access to string data");
            }
            return string_data[row_idx];
        }
        
        std::chrono::system_clock::time_point getDate(size_t row_idx) const {
            if (type != ColumnType::DATE || row_idx >= num_rows) {
                throw std::out_of_range("Invalid access to date data");
            }
            return date_data[row_idx];
        }
        
        // GPU operations (stubs to be implemented with actual CUDA code)
        bool transferToGPU() {
            // To be implemented with CUDA
            // This would allocate GPU memory and copy data to the GPU
            if (on_gpu) return true;  // Already on GPU
            
            on_gpu = true;  // Set this to true when implemented
            return on_gpu;
        }
        
        void freeGpuMemory() {
            // To be implemented with CUDA
            // Free the GPU memory if allocated
            if (on_gpu && gpu_data_ptr) {
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
    