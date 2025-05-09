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
enum class ComparisonOperator
{
    EQUALS,
    NOT_EQUALS,
    LESS_THAN,
    LESS_THAN_EQUALS,
    GREATER_THAN,
    GREATER_THAN_EQUALS
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
struct JoinCondition
{
    int leftColumnIdx;
    int rightColumnIdx;
    ComparisonOperator op;
    ColumnType columnType;
};
#endif // DEVICE_STRUCT_H