#include "enums.h"
#include <cstdint>

size_t sizeFromColumnType(ColumnType type){
    switch (type)
    {
    case ColumnType::STRING:
        return 0;
    case ColumnType::INT:
        return sizeof(float);
    case ColumnType::FLOAT:
        return sizeof(float);
    case ColumnType::DATE:
        return sizeof(int64_t); // Assuming date is stored as int64_t (nanoseconds since epoch)
    default:
        return 0;
    }
}
