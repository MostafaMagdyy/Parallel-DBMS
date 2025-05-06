#include "aggregate.h"
#include <iostream>
#include <utility>
#include <string>

AggregateFunctionType parseAggregateExpression(const std::string &name) {
    size_t openParen = name.find('(');
    size_t closeParen = name.find(')', openParen + 1);

    if (openParen == std::string::npos || closeParen == std::string::npos) {
        throw "Invalid aggregate expression format";
    }

    std::string functionName = name.substr(0, openParen);
    
    AggregateFunctionType function;
    if (functionName == "sum") {
        function = AggregateFunctionType::SUM;
    } else if (functionName == "avg") {
        function = AggregateFunctionType::AVG;
    } else if (functionName == "count") {
        function = AggregateFunctionType::COUNT;
    } else if (functionName == "min") {
        function = AggregateFunctionType::MIN;
    } else if (functionName == "max") {
        function = AggregateFunctionType::MAX;
    } else {
        throw "Unknown aggregate function: " + functionName;
    }
    return function;
    
}

