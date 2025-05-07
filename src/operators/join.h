#ifndef JOIN_H
#define JOIN_H
#include <cstddef>
#include "../headers/enums.h"
#include "../headers/table.h"

struct JoinCondition{
    size_t leftColumnIdx;
    size_t rightColumnIdx;
    ComparisonOperator op;
    ColumnType columnType;
};

void joinTablesCPU(std::shared_ptr<Table> left_table, std::shared_ptr<Table> right_table, std::vector<JoinCondition> join_conditions, std::shared_ptr<Table> result_table);



#endif