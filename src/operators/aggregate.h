#ifndef AGGREGATE_H
#define AGGREGATE_H

#include "operator_enums.h"
#include <utility>
#include <vector>
#include "../headers/table.h"


AggregateFunctionType parseAggregateExpression(const std::string &name);
void aggregateCPU(std::shared_ptr<Table> &table, std::vector<AggregateFunctionType> &aggregate_functions, std::vector<std::string> &column_names);

std::vector<void*> aggregate(std::shared_ptr<Table> table, std::vector<AggregateFunctionType> aggregate_functions, std::vector<std::string> column_names);

#endif // AGGREGATE_H