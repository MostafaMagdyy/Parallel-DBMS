#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <memory>
// --------duckdb includes----------------
#include <duckdb.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/execution/physical_operator.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/execution/operator/filter/physical_filter.hpp>
#include <duckdb/execution/operator/projection/physical_projection.hpp>
#include <duckdb/execution/operator/join/physical_join.hpp>
#include <duckdb/execution/operator/join/physical_hash_join.hpp>
#include <duckdb/execution/operator/join/physical_nested_loop_join.hpp>
#include <duckdb/execution/operator/join/physical_blockwise_nl_join.hpp>
#include <duckdb/execution/operator/order/physical_order.hpp>
#include <duckdb/execution/operator/order/physical_top_n.hpp>
#include <duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp>
#include <duckdb/parser/expression/constant_expression.hpp>
#include <duckdb/parser/expression/comparison_expression.hpp>
#include <duckdb/parser/expression/conjunction_expression.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <duckdb/optimizer/optimizer.hpp>

#include <sys/resource.h>
#include <cuda_runtime.h>
#include "headers/column.h"
#include "headers/table.h"
#include "headers/duckdb_manager.h"
#include "headers/enums.h"
#include "operators/operator_enums.h"
#include "operators/aggregate.h"
#include "operators/join.h"
#include "operators/sort.h"
#include "operators/cpu_sort.h"
#include "headers/constants.h"

namespace fs = std::filesystem;
#define OUTPUT_DIR "./output/"
bool use_gpu = true;

void createOutputDir()
{
    try
    {
        if (!fs::exists(OUTPUT_DIR))
        {
            std::cout << "Creating output directory: " << OUTPUT_DIR << std::endl;
            fs::create_directories(OUTPUT_DIR);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
    }
}

std::shared_ptr<Table> table_scan(DuckDBManager &manager, duckdb::PhysicalOperator *op, std::string indent)
{
    auto scan = reinterpret_cast<duckdb::PhysicalTableScan *>(op);
    auto params = scan->ParamsToString();
    if (params.find("Table") != params.end())
    {
        std::cout << indent << "Table Name: " << params["Table"] << std::endl;
    }
    // need further checks
    auto table = manager.getTable(params["Table"]);
    if (params.find("Table") != params.end() && scan->table_filters && !scan->table_filters->filters.empty())
    {
        table->clearFilters();
        std::vector<FilterCondition> filter_conditions;
        for (auto &kv : scan->table_filters->filters)
        {

            auto column_index = scan->column_ids[kv.first].GetPrimaryIndex();
            auto &filter = kv.second;
            std::string column_name;
            if (column_index < scan->names.size())
                column_name = scan->names[column_index];
            else
            {
                std::cerr << "Column index out of range: " << column_index << std::endl;
                continue;
            }
            if (filter->filter_type == duckdb::TableFilterType::CONSTANT_COMPARISON)
            {
                auto comparison = reinterpret_cast<duckdb::ConstantFilter *>(filter.get());
                // Map comparison type to filter operator
                ComparisonOperator op;
                switch (comparison->comparison_type)
                {
                case duckdb::ExpressionType::COMPARE_EQUAL:
                    op = ComparisonOperator::EQUALS;
                    break;
                case duckdb::ExpressionType::COMPARE_NOTEQUAL:
                    op = ComparisonOperator::NOT_EQUALS;
                    break;
                case duckdb::ExpressionType::COMPARE_LESSTHAN:
                    op = ComparisonOperator::LESS_THAN;
                    break;
                case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
                    op = ComparisonOperator::LESS_THAN_EQUALS;
                    break;
                case duckdb::ExpressionType::COMPARE_GREATERTHAN:
                    op = ComparisonOperator::GREATER_THAN;
                    break;
                case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                    op = ComparisonOperator::GREATER_THAN_EQUALS;
                    break;
                default:
                    continue;
                }

                const auto &columns = table->getColumns();
                if (column_index < columns.size())
                {
                    const auto &target_col = columns[column_index];
                    FilterCondition::FilterValue filter_value;
                    switch (target_col.type)
                    {
                    case ColumnType::FLOAT:
                        filter_value = comparison->constant.GetValue<float>();
                        break;
                    case ColumnType::STRING:
                        filter_value = comparison->constant.GetValue<std::string>();
                        break;
                    case ColumnType::DATE:
                        try
                        {
                            auto timestamp_str = comparison->constant.ToString();
                            filter_value = table->parseDate(timestamp_str);
                        }
                        catch (const std::exception &e)
                        {
                            std::cerr << "Failed to parse date from filter: " << e.what() << std::endl;
                            continue;
                        }
                        break;
                    default:
                        continue;
                    }

                    filter_conditions.emplace_back(column_name, op, filter_value);
                }
            }
            else if (filter->filter_type == duckdb::TableFilterType::CONJUNCTION_OR)
            {
                throw "CONJUCTION_OR not implemented";
            }
            else if (filter->filter_type == duckdb::TableFilterType::CONJUNCTION_AND)
            {
                throw "CONJUCTION_AND not implemented";
            }
            else
            {
                throw "Filter type not implemented";
            }
        }
        if (!filter_conditions.empty())
        {
            for (auto &filter : filter_conditions)
            {
                std::cout << indent << "Filter: " << filter.toString() << std::endl;
            }
            table->addFilters(filter_conditions);
        }
    }
    std::cout << indent << "  Projected columns: ";
    std::vector<std::size_t> projected_ids;
    for (size_t i = 0; i < scan->projection_ids.size(); i++)
    {
        auto proj_id = scan->projection_ids[i];
        proj_id = scan->column_ids[proj_id].GetPrimaryIndex();
        if (proj_id < scan->names.size())
        {

            std::cout << proj_id << ":" << scan->names[proj_id] << ' ';
            projected_ids.push_back(proj_id);
        }
        else
            std::cout << proj_id << "not found";
    }
    std::cout << std::endl;
    table->addProjectedColumns(projected_ids);
    if (!params.empty())
    {
        std::cout << indent << "Parameters:" << std::endl;
        for (const auto &pair : params)
        {
            std::cout << indent << "  " << pair.first << ": " << pair.second << std::endl;
        }
    }
    return table;
}
std::shared_ptr<Table> filter(DuckDBManager &manager, std::shared_ptr<Table> table, duckdb::PhysicalOperator *op, std::string indent)
{
    // TODO: Create new table with filtered data and return it
    auto filter = reinterpret_cast<duckdb::PhysicalFilter *>(op);
    std::cout << indent << "Filter Expression: " << filter->expression->ToString() << std::endl;
    // Add more detailed information about the filter if needed
    if (filter->expression)
        std::cout << filter->expression->ToString() << std::endl;

    std::cout << "Filter not implemented, returning original table" << std::endl;
    return table;
}

void print_expression(duckdb::Expression *expr, int indent_level = 0)
{
    if (!expr)
    {
        return;
    }

    std::string indent(indent_level * 4, ' ');
    std::cout << indent << "Expression Type: " << duckdb::ExpressionTypeToString(expr->GetExpressionType()) << std::endl;

    // // switch (expr->GetExpressionType()) {
    // //     case duckdb::ExpressionType::COLUMN_REF: {
    // //         auto col_ref = reinterpret_cast<duckdb::ColumnRefExpression *>(expr);
    // //         std::cout << indent << "  Column Name: " << col_ref->GetName() << std::endl;
    // //         break;
    // //     }
    // //     case duckdb::ExpressionType::VALUE_CONSTANT: {
    // //         auto const_expr = reinterpret_cast<duckdb::ConstantExpression *>(expr);
    // //         std::cout << indent << "  Value: " << const_expr->ToString() << std::endl;
    // //         break;
    // //     }
    // //     case duckdb::ExpressionType::COMPARE_EQUAL:
    // //     case duckdb::ExpressionType::COMPARE_GREATERTHAN:
    // //     case duckdb::ExpressionType::COMPARE_LESSTHAN:
    // //     case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
    // //     case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
    // //     case duckdb::ExpressionType::COMPARE_NOTEQUAL: {
    // //         auto comparison_expr = reinterpret_cast<duckdb::ComparisonExpression *>(expr);
    // //         std::cout << indent << "  Left Child:" << std::endl;
    // //         print_expression(comparison_expr->left.get(), indent_level + 1);
    // //         std::cout << indent << "  Right Child:" << std::endl;
    // //         print_expression(comparison_expr->right.get(), indent_level + 1);
    // //         break;
    // //     }
    // //     case duckdb::ExpressionType::CONJUNCTION_AND:
    // //     case duckdb::ExpressionType::CONJUNCTION_OR: {
    // //         auto logical_expr = reinterpret_cast<duckdb::ConjunctionExpression *>(expr);
    // //         for (size_t i = 0; i < logical_expr->children.size(); ++i) {
    // //             std::cout << indent << "  Child " << i + 1 << ":" << std::endl;
    // //             print_expression(logical_expr->children[i].get(), indent_level + 1);
    // //         }
    // //         break;
    // //     }
    // //     // Add cases for other expression types as needed
    // //     default:
    // //         // For expressions with children, recursively print them
    // //         // for (size_t i = 0; i < expr->child_count(); ++i) {
    // //         //     std::cout << indent << "  Child " << i + 1 << ":" << std::endl;
    // //         //     print_expression(expr->child(i).get(), indent_level + 1);
    // //         // }

    // //         std::cout << indent << "  (Other expression type, no further details available)" << std::endl;
    // //         break;
    // }
}
std::vector<std::string> getColumnNamesFromProjection(const duckdb::PhysicalOperator *op)
{
    std::vector<std::string> column_names;
    if (!op || op->children.empty())
    {
        return column_names;
    }
    duckdb::PhysicalOperator *child_op = &op->children[0].get();
    if (child_op && (child_op->type == duckdb::PhysicalOperatorType::PROJECTION ||
                     child_op->GetName() == "PROJECTION"))
    {
        auto projection = reinterpret_cast<duckdb::PhysicalProjection *>(child_op);
        for (size_t i = 0; i < projection->select_list.size(); i++)
        {
            column_names.push_back(projection->select_list[i]->ToString());
        }
    }
    else if (child_op)
    {
        std::cout << "ERROR: Child operator is not a projection" << std::endl;
    }

    return column_names;
}

std::shared_ptr<Table> nested_loop_join(DuckDBManager &manager, std::shared_ptr<Table> left, std::shared_ptr<Table> right, duckdb::PhysicalOperator *op, std::string indent)
{
    // TODO implement join and return the new made table
    auto join = reinterpret_cast<duckdb::PhysicalJoin *>(op);
    std::cout << indent << "Join Type: " << duckdb::JoinTypeToString(join->join_type) << std::endl;

    auto nested_join = reinterpret_cast<duckdb::PhysicalComparisonJoin *>(op);

    // Print join conditions
    std::cout << indent << "Join Conditions: ";

    std::vector<ColumnMetadata> result_columns;
    {
        std::vector<ColumnMetadata> left_columns = left->getColumns();
        std::vector<size_t> left_projected_column_indices = left->getProjectedColumnIndices();
        for (size_t i = 0; i < left_projected_column_indices.size(); i++)
        {
            std::cout << "left column: " << left_columns[left_projected_column_indices[i]].name << std::endl;
            result_columns.push_back(left_columns[left_projected_column_indices[i]]);
        }
    }
    {
        std::vector<ColumnMetadata> right_columns = right->getColumns();
        std::vector<size_t> right_projected_column_indices = right->getProjectedColumnIndices();
        for (size_t i = 0; i < right_projected_column_indices.size(); i++)
        {
            std::cout << "right column: " << right_columns[right_projected_column_indices[i]].name << std::endl;
            result_columns.push_back(right_columns[right_projected_column_indices[i]]);
        }
    }

    std::string result_table_name = left->getName() + "_" + right->getName() + "_join" + std::to_string(time(0));
    std::string result_table_path = OUTPUT_DIR + result_table_name + ".csv";

    for (size_t i = 0; i < result_columns.size(); i++)
    {
        std::cout << "result column: " << result_columns[i].name << std::endl;
    }
    std::shared_ptr<Table> result_table = std::make_shared<Table>(result_table_name, result_columns, result_table_path);
    result_table->createCSVHeaders();
    std::cout << "111111111" << std::endl;
    manager.addTable(result_table);
    std::cout << "222222222" << std::endl;
    std::vector<JoinCondition> join_conditions;
    std::vector<ColumnMetadata> left_columns = left->getColumns();
    bool string_join = false;
    for (size_t i = 0; i < nested_join->conditions.size(); i++)
    {
        if (left_columns[left->getColumnIndexOriginal(nested_join->conditions[i].left->ToString())].type == ColumnType::STRING)
            string_join = true;

        JoinCondition join_condition;
        join_condition.leftColumnIdx = left->getColumnIndexProjected(nested_join->conditions[i].left->ToString());
        join_condition.rightColumnIdx = right->getColumnIndexProjected(nested_join->conditions[i].right->ToString());
        join_condition.op = duckDBExpressionTypeToComparisonOperator(nested_join->conditions[i].comparison);
        join_condition.columnType = left_columns[left->getColumnIndexOriginal(nested_join->conditions[i].left->ToString())].type;
        join_conditions.push_back(join_condition);
    }
    std::cout << "333333333" << std::endl;
    for (size_t i = 0; i < join_conditions.size(); i++)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << left->getColumnName(join_conditions[i].leftColumnIdx) << " " << comparisonOperatorToString(join_conditions[i].op) << " " << right->getColumnName(join_conditions[i].rightColumnIdx) << std::endl;
    }
    std::cout << "444444444" << std::endl;
    std::cout << std::endl;
    if (use_gpu && !string_join)
    {
        std::cout << "using gpu" << std::endl;
        joinTablesGPU(left, right, join_conditions, result_table);
    }
    else
    {
        std::cout << "using cpu" << std::endl;
        joinTablesCPU(left, right, join_conditions, result_table);
    }

    result_table->saveCurrentBatch();
    return result_table;
}

std::shared_ptr<Table> projection(DuckDBManager &manager, std::shared_ptr<Table> table, duckdb::PhysicalOperator *op, std::string indent)
{
    // TODO: implement projection
    auto projection = reinterpret_cast<duckdb::PhysicalProjection *>(op);
    std::cout << indent << "Projection Expressions: ";
    std::vector<size_t> projected_columns;
    std::unordered_set<std::size_t> oringial_columns_set;
    for (size_t i = 0; i < projection->select_list.size(); i++)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << projection->select_list[i]->ToString() << ' ';
        std::size_t original_column_index = table->getColumnIndexOriginal(projection->select_list[i]->ToString());
        if (oringial_columns_set.find(original_column_index) == oringial_columns_set.end())
        {
            projected_columns.push_back(original_column_index);
            oringial_columns_set.insert(original_column_index);
        }
    }

    table->addProjectedColumns(projected_columns);
    std::cout << "Projection filter not implemented, original table returned" << std::endl;
    std::cout << std::endl;
    table->resetFilePositionToStart();
    return table;
}

std::shared_ptr<Table> order_by(DuckDBManager &manager, std::shared_ptr<Table> table, duckdb::PhysicalOperator *op, std::string indent)
{
    // TODO implement order by
    auto order = reinterpret_cast<duckdb::PhysicalOrder *>(op);
    std::cout << indent << "Order By: ";
    std::cout << "order->orders[0].expression->ToString(): " << order->orders[0].expression->ToString() << std::endl;
    std::string column_name = order->orders[0].expression->ToString();

    // Extract just the column name after the last dot
    size_t last_dot = column_name.find_last_of('.');
    if (last_dot != std::string::npos)
    {
        column_name = column_name.substr(last_dot + 1);
    }

    std::cout << "does it have quotes: " << (column_name[0] == '"' && column_name[column_name.size() - 1] == '"') << std::endl;
    std::cout << column_name[0] << ' ' << column_name[column_name.size() - 1] << std::endl;
    if (column_name[0] == '"' && column_name[column_name.size() - 1] == '"')
        column_name = column_name.substr(1, column_name.size() - 2);

    if (column_name[0] == '#')
    {
        int colIdx = std::stoi(column_name.substr(1));
        column_name = table->getColumnName(table->getProjectedColumnIndices()[colIdx]);
    }

    std::cout << "column_name: " << column_name << std::endl;
    std::cout << "table->getColumnType(column_name): " << columnTypeToString(table->getColumnType(column_name)) << std::endl;
    if (table->getColumnType(column_name) == ColumnType::STRING)
    {
        std::cout << "String column detected, using CPU sort" << std::endl;
        table->readNextBatch();
        sortTablesCPU(table, column_name);
        table->setIsResultTable(true);
        order->orders[0].type == duckdb::OrderType::DESCENDING ? table->setIsDescending(true) : table->setIsDescending(false);
        return table;
    }

    std::cout << "1111111111" << std::endl;
    std::vector<size_t> projected_column_indices = table->getProjectedColumnIndices();
    for (size_t i = 0; i < projected_column_indices.size(); i++)
    {
        std::cout << "projected column: " << table->getColumnName(projected_column_indices[i]) << std::endl;
    }
    int colIdx = table->getColumnIndexProjected(column_name);
    std::cout << "colIdx: " << colIdx << std::endl;
    std::cout << "2222222222" << std::endl;
    table->readNextBatch();
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    std::cout << "3333333333" << std::endl;
    auto current_batch = table->getCurrentBatch();
    std::cout << "current batch size: " << current_batch[0]->size() << std::endl;

    std::cout << "4444444444" << std::endl;
    std::vector<DeviceStruct> host_structs_in = table->transferBatchToGPU();
    std::vector<DeviceStruct> host_structs_out = table->createSortStructs();
    std::cout << "size of in: " << host_structs_in.size() << std::endl;
    std::cout << "size of out: " << host_structs_out.size() << std::endl;
    std::cout << "7777777777" << std::endl;
    DeviceStruct *device_structs_in;
    DeviceStruct *device_structs_out;
    cudaMalloc(&device_structs_in, host_structs_in.size() * sizeof(DeviceStruct));
    cudaMalloc(&device_structs_out, host_structs_out.size() * sizeof(DeviceStruct));
    cudaMemcpy(device_structs_in, host_structs_in.data(), host_structs_in.size() * sizeof(DeviceStruct), cudaMemcpyHostToDevice);
    cudaMemcpy(device_structs_out, host_structs_out.data(), host_structs_out.size() * sizeof(DeviceStruct), cudaMemcpyHostToDevice);

    std::cout << "8888888888" << std::endl;
    std::cout << "column size: " << current_batch[colIdx]->size() << std::endl;
    std::cout << "radix sort started" << std::endl;
    radix_sort(device_structs_out, device_structs_in, current_batch[0]->getNumRows(), colIdx, table->getProjectedColumnIndices().size() + 1);

    std::cout << "9999999999" << std::endl;
    // Create a vector of DeviceStruct objects instead of pointers
    std::vector<DeviceStruct> h_final_in(host_structs_in.size());
    cudaMemcpy(h_final_in.data(), device_structs_in, host_structs_in.size() * sizeof(DeviceStruct), cudaMemcpyDeviceToHost);
    std::vector<void *> results;
    for (size_t i = 0; i < h_final_in.size(); i++)
    {
        std::cout << "123123123123" << std::endl;
        switch (h_final_in[i].type) // Use . instead of ->
        {
        case ColumnType::FLOAT:
        {
            std::cout << "in" << std::endl;
            void *result_data = malloc(h_final_in[i].numRows * sizeof(float));
            cudaMemcpy(result_data, h_final_in[i].device_ptr, h_final_in[i].numRows * sizeof(float), cudaMemcpyDeviceToHost);
            results.push_back(result_data);
            break;
        }
        case ColumnType::DATE:
        {
            std::cout << "in" << std::endl;
            void *result_data = malloc(h_final_in[i].numRows * sizeof(int64_t));
            cudaMemcpy(result_data, h_final_in[i].device_ptr, h_final_in[i].numRows * sizeof(int64_t), cudaMemcpyDeviceToHost);
            results.push_back(result_data);
            break;
        }
        case ColumnType::STRING:
        {
            std::cout << "in" << std::endl;
            void *result_data = malloc(h_final_in[i].numRows * MAX_STRING_LENGTH);
            cudaMemcpy(result_data, h_final_in[i].device_ptr, h_final_in[i].numRows * MAX_STRING_LENGTH, cudaMemcpyDeviceToHost);
            results.push_back(result_data);
            break;
        }
        }
    }

    std::vector<ColumnMetadata> result_columns;
    for (size_t i = 0; i < table->getProjectedColumnIndices().size(); i++)
    {
        result_columns.push_back(table->getColumns()[table->getProjectedColumnIndices()[i]]);
        std::cout << "result column: " << result_columns[i].name << std::endl;
    }
    std::string result_table_name = table->getName() + "_ordered" + std::to_string(time(0));
    std::string result_table_path = "./temp_csv/" + result_table_name + ".csv";
    std::shared_ptr<Table> result_table = std::make_shared<Table>(result_table_name, result_columns, result_table_path);
    manager.addTable(result_table);
    std::cout << "h_final_in[0].numRows: " << h_final_in[0].numRows << std::endl;
    result_table->addResultBatch(results.data(), h_final_in[0].numRows);
    result_table->setSaveFilePath(result_table_path);
    cudaFree(device_structs_in);
    cudaFree(device_structs_out);
    for (size_t i = 0; i < host_structs_in.size(); i++)
    {
        DeviceStruct::deleteStruct(host_structs_in[i]);
    }
    for (size_t i = 0; i < host_structs_out.size(); i++)
    {
        DeviceStruct::deleteStruct(host_structs_out[i]);
    }

    order->orders[0].type == duckdb::OrderType::DESCENDING ? result_table->setIsDescending(true) : result_table->setIsDescending(false);

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << "total time taken by order by: " << duration.count() * 1000 << " ms" << std::endl;
    result_table->setIsResultTable(true);
    return result_table;
}

std::shared_ptr<Table> aggregate(DuckDBManager &manager, std::shared_ptr<Table> table, duckdb::PhysicalOperator *op, std::string indent)
{

    // TODO implement aggreagate on the passed table corrrectly.
    auto aggregate_op = reinterpret_cast<duckdb::PhysicalUngroupedAggregate *>(op);

    std::vector<std::string> column_names = getColumnNamesFromProjection(op);
    std::cout << indent << "Aggregate Functions: ";

    std::vector<AggregateFunctionType> aggregate_functions;
    std::vector<AggregateFunctionType> aggregate_functions_temp;
    for (size_t i = 0; i < aggregate_op->aggregates.size(); i++)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << aggregate_op->aggregates[i]->ToString();
        AggregateFunctionType aggFunc = parseAggregateExpression(aggregate_op->aggregates[i]->ToString()); // we will assume that index #0, #1 is always ordered so we don't have to parse it ourselves
        if (aggFunc == AggregateFunctionType::AVG)
            aggregate_functions.push_back(AggregateFunctionType::SUM);
        else
            aggregate_functions.push_back(aggFunc);

        aggregate_functions_temp.push_back(aggFunc);
        std::cout << "Aggregate Function: " << aggregateFunctionTypeToString(aggFunc) << " Column Name " << column_names[i] << std::endl;
    }
    std::vector<ColumnMetadata> columns = table->getColumns();

    for (auto &col : columns)
    {
        std::cout << "Column name: " << col.name << std::endl;
    }
    std::vector<void *> results = aggregate(table, aggregate_functions, column_names);
    std::vector<ColumnMetadata> result_columns;
    std::vector<std::string> result_column_names;
    for (size_t i = 0; i < column_names.size(); i++)
    {
        result_column_names.push_back(aggregateFunctionTypeToString(aggregate_functions_temp[i]) + "(" + column_names[i] + ")");
    }
    for (size_t i = 0; i < results.size(); i++)
    {
        std::string duckdb_type = "";
        bool is_primary_key = false;
        size_t element_size = 0;

        ColumnMetadata column(result_column_names[i],
                              columns[table->getColumnIndexOriginal(column_names[i])].type,
                              duckdb_type,
                              is_primary_key,
                              i);
        result_columns.push_back(column);
    }
    for (int i = 0; i < aggregate_functions_temp.size(); i++)
    {
        if (aggregate_functions_temp[i] == AggregateFunctionType::AVG)
        {
            size_t total_count = table->getTotalCount();
            std::cout << "From AVG: " << total_count << std::endl;
            float value = *(float *)results[i];
            value = value / total_count;
            *(float *)results[i] = value;
        }
    }

    std::string result_table_name = table->getName() + "_agg" + std::to_string(time(0));
    std::string result_table_path = OUTPUT_DIR + result_table_name + ".csv";
    std::shared_ptr<Table> result_table = std::make_shared<Table>(result_table_name, result_columns, result_table_path);
    result_table->createCSVHeaders();
    result_table->addResultBatch(results.data(), 1);
    result_table->saveCurrentBatch();
    manager.addTable(result_table);
    return result_table;
}

std::shared_ptr<Table> traversePhysicalOperator(DuckDBManager &manager, duckdb::PhysicalOperator *op, int depth = 0)
{
    if (!op)
        return nullptr;

    // Print indentation based on depth
    std::string indent(depth * 2, ' ');
    std::vector<std::shared_ptr<Table>> child_tables; // Assuming at most two children for simplicity
    for (auto &child : op->children)
        child_tables.push_back(traversePhysicalOperator(manager, &child.get(), depth + 1));

    // Print information about the current operator
    std::cout << indent << "Operator Type: " << op->GetName() << std::endl;
    std::shared_ptr<Table> result_table = nullptr;
    switch (op->type)
    {
    case duckdb::PhysicalOperatorType::TABLE_SCAN:
        result_table = table_scan(manager, op, indent);
        break;

    case duckdb::PhysicalOperatorType::FILTER:
        result_table = filter(manager, child_tables[0], op, indent);
        break;

    case duckdb::PhysicalOperatorType::PROJECTION:
        result_table = projection(manager, child_tables[0], op, indent);
        break;

    case duckdb::PhysicalOperatorType::UNGROUPED_AGGREGATE:
        result_table = aggregate(manager, child_tables[0], op, indent);
        break;

    case duckdb::PhysicalOperatorType::BLOCKWISE_NL_JOIN:
        std::cout << "BlockWise nl not implemented not implemented" << std::endl;
        break;
    case duckdb::PhysicalOperatorType::HASH_JOIN:
    case duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN:
        if (child_tables.size() < 2)
        {
            std::cerr << "ERROR: Join operator has less than 2 child tables." << std::endl;
            return nullptr;
        }
        result_table = nested_loop_join(manager, child_tables[0], child_tables[1], op, indent);
        break;

    case duckdb::PhysicalOperatorType::ORDER_BY:
        // std::string table_name=manager.getTable(params["Table"]).getName();
        result_table = order_by(manager, child_tables[0], op, indent);
        break;

    default:
        break;
    }
    std::cout << indent << "------------------------" << std::endl;
    return result_table;
}
std::vector<std::string> readQueries(std::string queries_dir)
{
    std::vector<std::string> test_queries;
    if (fs::exists(queries_dir) && fs::is_directory(queries_dir))
    {
        std::cout << "Reading queries from " << queries_dir << " directory..." << std::endl;
        for (const auto &entry : fs::directory_iterator(queries_dir))
        {
            if (entry.is_regular_file())
            {
                std::ifstream query_file(entry.path());
                if (query_file.is_open())
                {
                    std::string query_text;
                    std::string line;
                    while (std::getline(query_file, line))
                    {
                        query_text += line + " ";
                    }
                    if (!query_text.empty())
                    {
                        test_queries.push_back(query_text);
                        std::cout << "Loaded query from " << entry.path().filename() << std::endl;
                    }
                    query_file.close();
                }
                else
                {
                    std::cerr << "Failed to open " << entry.path() << std::endl;
                }
            }
        }
    }
    else
    {
        std::cerr << "Test queries directory not found: " << queries_dir << std::endl;
        return {};
    }
    return test_queries;
}

int main(int argc, char *argv[])
{
    // if (argc != 3) {
    //     std::cerr << "Usage: " << argv[0]
    //               << " <csv_directory> \"<SQL query>\"\n";
    //     return 1;
    // }
    const std::string csv_directory = "./csv_data";
    const std::string query = argv[1];
    createOutputDir();

    if (argc > 2)
    {
        use_gpu = std::stoi(argv[2]) == 1;
    }

    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    try
    {
        // 1) initialize schemas from CSV files
        auto db_manager = DuckDBManager::create();
        db_manager.initializeTablesFromCSVs(csv_directory);
        db_manager.listAllTables();

        // 2) run exactly the one query
        std::cout << "\nPlanning query: " << query << "\n\n";

        // default plan
        std::cout << "Default plan visualization:\n";
        auto plan = db_manager.getQueryPlan(query);
        std::cout << std::endl;

        // custom physical traversal & execution
        std::cout << "Custom tree traversal / execution:\n";
        auto result_table = traversePhysicalOperator(db_manager, plan);
        if (plan->type == duckdb::PhysicalOperatorType::PROJECTION || plan->type == duckdb::PhysicalOperatorType::ORDER_BY || plan->type == duckdb::PhysicalOperatorType::TABLE_SCAN)
        {
            const auto out_path = OUTPUT_DIR + result_table->getName() + "_result.csv";
            result_table->setSaveFilePath(out_path);
            result_table->resetFilePositionToStart();
            result_table->createCSVHeaders();
            while (result_table->hasMoreData())
            {
                if (!result_table->getIsResultTable())
                    result_table->readNextBatch();
                result_table->saveCurrentBatch();
                if (result_table->getIsResultTable())
                    break;
            }
            std::cout << std::endl;
        }
        std::cout << "=========================================" << std::endl;
        // std::cout << db_manager.readNextBatch("employees") << std::endl;
        // db_manager.printCurrentBatch("employees", 10, 30);

        std::shared_ptr<Table> table = db_manager.getTable("employees");
        std::cout << table->getColumns().size() << '\n';
        std::cout << table->getProjectedColumnNames().size() << '\n';
        std::cout << "=========================================" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time)
            .count();
    std::cout << "Total time: " << total_s << "s\n";
    return 0;
}
