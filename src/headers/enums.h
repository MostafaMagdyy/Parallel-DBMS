#ifndef ENUMS_H
#define ENUMS_H
#include <iostream>
#include <cstddef>
#include <string>   
#include "duckdb/common/enums/expression_type.hpp" 

enum class ColumnType
{
    STRING, // VARCHAR/TEXT
    FLOAT, // NUMERIC/FLOAT
    DATE,   // TIMESTAMP
    UNKNOWN
};

size_t sizeFromColumnType(ColumnType type);
std::string columnTypeToString(ColumnType type);

enum class ComparisonOperator {
    EQUALS,
    NOT_EQUALS,
    LESS_THAN,
    LESS_THAN_EQUALS,
    GREATER_THAN,
    GREATER_THAN_EQUALS
};

template<typename T>
bool compareValues(T a, T b, ComparisonOperator op) {
    // std :: cout << "comparing " << a << " and " << b << std :: endl; 
    switch(op) {
        case ComparisonOperator::EQUALS:
            return a == b;
        case ComparisonOperator::NOT_EQUALS:
            return a != b;
        case ComparisonOperator::LESS_THAN:
            return a < b;
        case ComparisonOperator::LESS_THAN_EQUALS:
            return a <= b;
        case ComparisonOperator::GREATER_THAN:
            return a > b;
        case ComparisonOperator::GREATER_THAN_EQUALS:
            return a >= b;  
        default:
            throw "Invalid comparison operator";    
    }
}

std::string comparisonOperatorToString(ComparisonOperator op);

ComparisonOperator duckDBExpressionTypeToComparisonOperator(duckdb::ExpressionType type);

ComparisonOperator getComparisonOperator(std::string op);

#endif // ENUMS_H