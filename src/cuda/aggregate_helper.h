#ifndef AGGREGATE_HELPERS_H
#define AGGREGATE_HELPERS_H

#include "aggregate.cuh"
#include "../headers/column.h"
#include "../headers/table.h"
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

// Convert ColumnType to ValueType
ValueType columnTypeToValueType(ColumnType col_type) {
    switch (col_type) {
        case ColumnType::FLOAT:
            return TYPE_FLOAT;
        case ColumnType::DATE:
            return TYPE_DATE;
        default:
            throw std::runtime_error("Unsupported column type for aggregation");
    }
}

// Format date from timestamp
std::string formatTimestampAsDate(int64_t timestamp) {
    time_t time_value = static_cast<time_t>(timestamp);
    struct tm tm_info;
    localtime_r(&time_value, &tm_info);
    
    std::ostringstream oss;
    oss << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Format aggregate result for display
std::string formatAggregateResult(const AggregateResult& result, ColumnType column_type, AggregateType agg_type) {
    std::ostringstream oss;
    
    if (column_type == ColumnType::FLOAT) {
        if (agg_type == AGG_COUNT) {
            oss << result.int_val;
        } else {
            oss << std::fixed << std::setprecision(6) << result.float_val;
        }
    }
    else if (column_type == ColumnType::DATE) {
        if (agg_type == AGG_MIN || agg_type == AGG_MAX) {
            oss << formatTimestampAsDate(result.int_val);
        } else if (agg_type == AGG_COUNT) {
            oss << result.int_val;
        } else if (agg_type == AGG_AVG) {
            oss << formatTimestampAsDate(result.int_val);
        } else {
            oss << result.int_val;
        }
    }
    
    return oss.str();
}

#endif