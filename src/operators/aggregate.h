#ifndef AGGREGATE_H
#define AGGREGATE_H

#include "operator_enums.h"
#include <utility>


AggregateFunctionType parseAggregateExpression(const std::string &name);

#endif // AGGREGATE_H