#ifndef DEVICE_STRUCT_H
#define DEVICE_STRUCT_H

#include <cuda_runtime.h>
#include "enums.h"

class DeviceStruct {
    ColumnType type; 
    void* device_ptr; // GPU memory pointer (to be used with CUDA APIs)
    int numRows;
    int rowSize;

    public:
    // Constructor declaration
    DeviceStruct(ColumnType type, void* host_ptr, int numRows);

    // Move constructor declaration
    DeviceStruct(DeviceStruct&& other) noexcept;
    // Destructor declaration
    ~DeviceStruct();
};

#endif // DEVICE_STRUCT_H