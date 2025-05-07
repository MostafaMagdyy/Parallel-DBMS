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

#include <chrono>
#include <sys/resource.h>
#include "headers/column.h"
#include "headers/table.h"
#include "headers/duckdb_manager.h"
#include "headers/enums.h"
#include "operators/operator_enums.h"
#include "operators/aggregate.h"
#include "operators/join.h"

namespace fs = std::filesystem;

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

        // Apply the filters to the table
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
    // Testing Aggregate ON GPU
    // if (table->getName() == "emplyees")
    // {
    //     table->readNextBatch();
    //     std::cout << "GPU STARTED" << std::endl;
    //     std::string result = table->computeAggregate("salary", AggregateType::AGG_MAX);
    //     std::cout << indent << "Aggregate Result: " << result << std::endl;
    //     std::cout << "GPU FINISHED" << std::endl;
    //     table->printCurrentBatch();
    // }

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
        for(size_t i = 0; i < left_projected_column_indices.size(); i++) {
            std::cout << "left column: " << left_columns[left_projected_column_indices[i]].name << std::endl;
            result_columns.push_back(left_columns[left_projected_column_indices[i]]);
        }
    }
    {
        std::vector<ColumnMetadata> right_columns = right->getColumns();
        std::vector<size_t> right_projected_column_indices = right->getProjectedColumnIndices();
        for(size_t i = 0; i < right_projected_column_indices.size(); i++) {
            std::cout << "right column: " << right_columns[right_projected_column_indices[i]].name << std::endl;
            result_columns.push_back(right_columns[right_projected_column_indices[i]]);
        }
    }   


    std::string result_table_name = left->getName() + "_" + right->getName() + "_join" + std::to_string(time(0));
    std::string result_table_path = "./temp_csv/" + result_table_name + ".csv";
    for(size_t i = 0; i < result_columns.size(); i++) {
        std::cout << "result column: " << result_columns[i].name << std::endl;
    }
    std::shared_ptr<Table> result_table = std::make_shared<Table>(result_table_name, result_columns, result_table_path);
    result_table->createCSVHeaders();
    std::cout << "111111111" << std::endl;
    manager.addTable(result_table);  
    std::cout << "222222222" << std::endl;
    std::vector<JoinCondition> join_conditions; 
    std::vector<ColumnMetadata> left_columns = left->getColumns();
    for (size_t i = 0; i < nested_join->conditions.size(); i++)
    {
        JoinCondition  join_condition;
        join_condition.leftColumnIdx = left->getColumnIndexProjected(nested_join->conditions[i].left->ToString());
        join_condition.rightColumnIdx = right->getColumnIndexProjected(nested_join->conditions[i].right->ToString());
        join_condition.op = duckDBExpressionTypeToComparisonOperator(nested_join->conditions[i].comparison);
        join_condition.columnType = left_columns[left->getColumnIndexOriginal(nested_join->conditions[i].left->ToString())].type;
        join_conditions.push_back(join_condition);      
    }
    std::cout << "333333333" << std::endl;
    for (size_t i = 0; i < join_conditions.size(); i++) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << left->getColumnName(join_conditions[i].leftColumnIdx) << " " << comparisonOperatorToString(join_conditions[i].op) << " " << right->getColumnName(join_conditions[i].rightColumnIdx) << std::endl;
    }
    std::cout << "444444444" << std::endl;
    std::cout << std::endl;

    joinTablesCPU(left, right, join_conditions, result_table);
    result_table->saveCurrentBatch();

    return result_table;
}

std::shared_ptr<Table> projection(DuckDBManager &manager, std::shared_ptr<Table> table, duckdb::PhysicalOperator *op, std::string indent)
{
    // TODO: implement projection
    auto projection = reinterpret_cast<duckdb::PhysicalProjection *>(op);
    std::cout << indent << "Projection Expressions: ";
    for (size_t i = 0; i < projection->select_list.size(); i++)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << projection->select_list[i]->ToString() << ' ';
    }
    std::cout << "Projection filter not implemented, original table returned" << std::endl;
    std::cout << std::endl;
    return table;
}

std::shared_ptr<Table> order_by(DuckDBManager &manager, std::shared_ptr<Table> table, duckdb::PhysicalOperator *op, std::string indent)
{
    // TODO implement order by
    std::shared_ptr<Table> result_table = table;

    auto order = reinterpret_cast<duckdb::PhysicalOrder *>(op);
    std::cout << indent << "Order By: ";
    for (size_t i = 0; i < order->orders.size(); i++)
    {
        if (i > 0)
            std::cout << ", ";
        // Extract column names for sorting from the order operation
        // if (!order_columns.empty() && i < order_columns.size()) {
        //     std::string full_expr = order_columns[i];
        //     std::string clean_column = full_expr;

        //     // Find the last dot and extract just the column name
        //     size_t last_dot_pos = full_expr.find_last_of('.');
        //     if (last_dot_pos != std::string::npos) {
        //         clean_column = full_expr.substr(last_dot_pos + 1);
        //     }

        //     std::cout << clean_column << " "
        //              << (order->orders[i].type == duckdb::OrderType::ASCENDING ? "ASC" : "DESC");
        // } else {
        //     std::cout << order->orders[i].expression->ToString() << " "
        //              << (order->orders[i].type == duckdb::OrderType::ASCENDING ? "ASC" : "DESC");
        // }
    }
    std::cout << "order by filter not implemented, original table returned" << std::endl;
    std::cout << std::endl;
    return result_table;
}

std::shared_ptr<Table> aggregate(DuckDBManager &manager, std::shared_ptr<Table> table, duckdb::PhysicalOperator *op, std::string indent)
{

    // TODO implement aggreagate on the passed table corrrectly.
    auto aggregate = reinterpret_cast<duckdb::PhysicalUngroupedAggregate *>(op);

    std::vector<std::string> column_names = getColumnNamesFromProjection(op);
    std::cout << indent << "Aggregate Functions: ";

    std::vector<AggregateFunctionType> aggregate_functions;

    for (size_t i = 0; i < aggregate->aggregates.size(); i++)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << aggregate->aggregates[i]->ToString();
        AggregateFunctionType aggFunc = parseAggregateExpression(aggregate->aggregates[i]->ToString()); // we will assume that index #0, #1 is always ordered so we don't have to parse it ourselves
        aggregate_functions.push_back(aggFunc);
        std::cout << "Aggregate Function: " << aggregateFunctionTypeToString(aggFunc) << " Column Name " << column_names[i] << std::endl;
    }

    std::cout << "Aggregate filter not implemented, original table returned" << std::endl;
    std::cout << std::endl;
    return table;
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
    try
    {
        auto db_manager = DuckDBManager::create();
        // Initialize tables from CSV files in a directory (schema only)
        std::string csv_directory = "./csv_data";
        db_manager.initializeTablesFromCSVs(csv_directory);
        db_manager.listAllTables();

        // Create a vector to store test queries

        // Read queries from files in test_queries directory
        std::string queries_dir = "./test_queries";
        std::vector<std::string> test_queries = readQueries(queries_dir);

        std::cout << "=========================================" << std::endl;
        // std::cout << "=========================================" << std::endl;
        // std::cout << db_manager.readNextBatch("employees") << std::endl;
        for (auto &query : test_queries)
        {
            std::cout << "\n=========================================" << std::endl;
            std::cout << "Planning query: " << query << std::endl;
            std::cout << "=========================================\n"
                      << std::endl;

            // Get the logical plan
            std::cout << "Default plan visualization:" << std::endl;
            auto plan = db_manager.getQueryPlan(query);
            // Print the default tree visualization
            // plan->Print();
            std::cout << std::endl;
            // Use our custom traversal function
            std::cout << "Custom tree traversal:" << std::endl;
            traversePhysicalOperator(db_manager, plan);
            std::cout << std::endl;
        }
        std::cout << "=========================================" << std::endl;
        // std::cout << db_manager.readNextBatch("employees") << std::endl;
        // db_manager.printCurrentBatch("employees", 10, 30);

        std::shared_ptr<Table> table = db_manager.getTable("employees");
        std::cout << table->getColumns().size() << '\n';
        std::cout << table->getProjectedColumnNames().size()<< '\n';
        std::cout << "=========================================" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}