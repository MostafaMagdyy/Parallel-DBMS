#include "enums.h"
#include <cstdint>
#include "constants.h"
size_t sizeFromColumnType(ColumnType type){
    switch (type)
    {
    case ColumnType::STRING:
        return MAX_STRING_LENGTH * sizeof(char);
    case ColumnType::FLOAT:
        return sizeof(float);
    case ColumnType::DATE:
        return sizeof(int64_t); // Assuming date is stored as int64_t (nanoseconds since epoch)
    default:
        return 0;
    }
}

std::string comparisonOperatorToString(ComparisonOperator op)
{
    switch (op)
    {
    case ComparisonOperator::EQUALS:
        return "=";
    case ComparisonOperator::NOT_EQUALS:
        return "!=";
    case ComparisonOperator::LESS_THAN:
        return "<";
    case ComparisonOperator::LESS_THAN_EQUALS:
        return "<=";
    case ComparisonOperator::GREATER_THAN:
        return ">";
    case ComparisonOperator::GREATER_THAN_EQUALS:
        return ">=";
    default:
        return "";
    }
}


ComparisonOperator duckDBExpressionTypeToComparisonOperator(duckdb::ExpressionType type)
{
    switch (type){
        case duckdb::ExpressionType::COMPARE_EQUAL:
            return ComparisonOperator::EQUALS;
        case duckdb::ExpressionType::COMPARE_NOTEQUAL:
            return ComparisonOperator::NOT_EQUALS;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
            return ComparisonOperator::LESS_THAN;   
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return ComparisonOperator::LESS_THAN_EQUALS;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
            return ComparisonOperator::GREATER_THAN;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return ComparisonOperator::GREATER_THAN_EQUALS; 
        default:
            throw "Invalid filter operator: " + duckdb::ExpressionTypeToString(type);   
    }
}

ComparisonOperator getComparisonOperator(std::string op)
{
    if(op == "=")
    {
        return ComparisonOperator::EQUALS;
    }
    else if(op == "!=")
    {
        return ComparisonOperator::NOT_EQUALS;
    }
    else if(op == "<")
    {
        return ComparisonOperator::LESS_THAN;
    }
    else if(op == "<=") 
    {
        return ComparisonOperator::LESS_THAN_EQUALS;
    }
    else if(op == ">")
    {
        return ComparisonOperator::GREATER_THAN;
    }
    else if(op == ">=")
    {
        return ComparisonOperator::GREATER_THAN_EQUALS;
    }
    else
    {
        throw "Invalid filter operator: " + op;
    }
}
