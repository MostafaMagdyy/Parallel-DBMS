#ifndef DEVICE_STRUCT_H
#define DEVICE_STRUCT_H

enum class ColumnType
{
    STRING,
    INT,
    FLOAT,
    DATE,
    UNKNOWN,
    UINT64,
};

class DeviceStruct
{

public:
    ColumnType type;
    void *device_ptr; // GPU memory pointer (to be used with CUDA APIs)
    int numRows;
    int rowSize;
    // Constructor declaration
    DeviceStruct()
    {
    }
    // DeviceStruct(ColumnType type, void *host_ptr, int numRows);
    // Move constructor declaration
    // DeviceStruct(DeviceStruct &&other) noexcept;
    // Destructor declaration
    ~DeviceStruct()
    {
    }
};

#endif // DEVICE_STRUCT_H