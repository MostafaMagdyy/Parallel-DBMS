#ifndef DEVICE_STRUCT_H
#define DEVICE_STRUCT_H

#include <cuda_runtime.h>
#include "enums.h"

struct DeviceStruct
{
    ColumnType type;
    void *device_ptr; // GPU memory pointer (to be used with CUDA APIs)
    size_t numRows;
    size_t rowSize;
    DeviceStruct(ColumnType type, void *device_ptr, size_t numRows, size_t rowSize);
    DeviceStruct() {};
    // Constructor declaration

    static DeviceStruct *createStruct(ColumnType type, void *host_ptr, size_t numRows);
    static DeviceStruct *createStructWithoutCopy(ColumnType type, size_t numRows);
    static void deleteStruct(DeviceStruct &deviceStruct);
    

    // Destructor declaration
    ~DeviceStruct();
};

#endif // DEVICE_STRUCT_H