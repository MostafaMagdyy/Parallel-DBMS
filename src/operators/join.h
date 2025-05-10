#ifndef JOIN_H__2
#define JOIN_H__2

#include "../headers/table.h"
#include "../headers/device_struct.h"
#include "../headers/enums.h"
#include <vector>

struct JoinCondition{
    size_t leftColumnIdx;
    size_t rightColumnIdx;
    ComparisonOperator op;
    ColumnType columnType;
};

void joinTablesCPU(std::shared_ptr<Table> left_table, std::shared_ptr<Table> right_table,
                   std::vector<JoinCondition> join_conditions,
                   std::shared_ptr<Table> result_table);

void joinTablesGPU(std::shared_ptr<Table> left_table, std::shared_ptr<Table> right_table,
                   std::vector<JoinCondition> join_conditions,
                   std::shared_ptr<Table> result_table);

#endif