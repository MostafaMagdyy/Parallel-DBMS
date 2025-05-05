#ifndef ENUMS_H
#define ENUMS_H

enum class ColumnType
{
    STRING, // VARCHAR/TEXT
    INT, // NUMERIC/FLOAT
    FLOAT, // NUMERIC/FLOAT
    DATE,   // TIMESTAMP
    UNKNOWN
};

int sizeFromColumnType(ColumnType type);


enum class FilterOperator {
    EQUALS,
    NOT_EQUALS,
    LESS_THAN,
    LESS_THAN_EQUALS,
    GREATER_THAN,
    GREATER_THAN_EQUALS
};

#endif // ENUMS_H