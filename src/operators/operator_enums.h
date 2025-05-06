#ifndef OPERATORENUMS_H
#define OPERATORENUMS_H

#include <string>

enum class AggregateFunctionType {
    SUM,
    AVG,
    COUNT,
    MIN,
    MAX
};


inline AggregateFunctionType getAggregateFunction(const std::string& func) {
    if (func == "SUM") return AggregateFunctionType::SUM;
    else if (func == "AVG") return AggregateFunctionType::AVG;
    else if (func == "COUNT") return AggregateFunctionType::COUNT;
    else if (func == "MIN") return AggregateFunctionType::MIN;
    else if (func == "MAX") return AggregateFunctionType::MAX;
    throw "Invalid aggregate function: " + func;
}

inline std::string aggregateFunctionTypeToString(AggregateFunctionType func) {
    switch (func) {
        case AggregateFunctionType::SUM: return "SUM";
        case AggregateFunctionType::AVG: return "AVG";
        case AggregateFunctionType::COUNT: return "COUNT";
        case AggregateFunctionType::MIN: return "MIN";
        case AggregateFunctionType::MAX: return "MAX";
        default: throw "Invalid aggregate function enum value.";
    }
}


#endif // ENUMS_H