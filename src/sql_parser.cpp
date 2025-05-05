#include <iostream>
#include <vector>
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
namespace fs = std::filesystem;

void table_scan(DuckDBManager &manager, duckdb::PhysicalOperator *op, std::string indent)
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
                FilterOperator op;
                switch (comparison->comparison_type)
                {
                case duckdb::ExpressionType::COMPARE_EQUAL:
                    op = FilterOperator::EQUALS;
                    break;
                case duckdb::ExpressionType::COMPARE_NOTEQUAL:
                    op = FilterOperator::NOT_EQUALS;
                    break;
                case duckdb::ExpressionType::COMPARE_LESSTHAN:
                    op = FilterOperator::LESS_THAN;
                    break;
                case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
                    op = FilterOperator::LESS_THAN_EQUALS;
                    break;
                case duckdb::ExpressionType::COMPARE_GREATERTHAN:
                    op = FilterOperator::GREATER_THAN;
                    break;
                case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
                    op = FilterOperator::GREATER_THAN_EQUALS;
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
            } else if (filter->filter_type == duckdb::TableFilterType::EXPRESSION_FILTER)
            {
                std::cout << indent << "Expression Filter: "  << std::endl;
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
    // if (table->getName() == "projects")
    // {
    //     table->readNextBatch();
    //     std::string result = table->computeAggregate("budget", AggregateType::AGG_MAX);
    //     std::cout << indent << "Aggregate Result: " << result << std::endl;
    //     std::cout << "GPU STARTED" << std::endl;
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

}


void filter(DuckDBManager &manager, duckdb::PhysicalOperator *op, std::string indent)
{
    auto filter = reinterpret_cast<duckdb::PhysicalFilter *>(op);
    std::cout << indent << "Filter Expression: " << filter->expression->ToString() << std::endl;
    // Add more detailed information about the filter if needed
    if (filter->expression)
    std::cout << filter->expression->ToString() << std::endl;
}


void print_expression(duckdb::Expression *expr, int indent_level = 0) {
    if (!expr) {
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

void nested_loop_join(DuckDBManager &manager, duckdb::PhysicalOperator *op, std::string indent)
{
    auto join = reinterpret_cast<duckdb::PhysicalJoin *>(op);
    std::cout << indent << "Join Type: " << duckdb::JoinTypeToString(join->join_type) << std::endl;
    if (op->type == duckdb::PhysicalOperatorType::HASH_JOIN)
    {
        auto hash_join = reinterpret_cast<duckdb::PhysicalHashJoin *>(op);

        // Print join conditions
        std::cout << indent << "Join Conditions: ";
        for (size_t i = 0; i < hash_join->conditions.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << hash_join->conditions[i].left->ToString() << " "
                      << duckdb::ExpressionTypeToOperator(hash_join->conditions[i].comparison) << " "
                      << hash_join->conditions[i].right->ToString();
        }
        std::cout << std::endl;
    }
    else if (op->type == duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN)
    {
        auto nested_join = reinterpret_cast<duckdb::PhysicalNestedLoopJoin *>(op);

        // Print join conditions
        std::cout << indent << "Join Conditions: ";
        for (size_t i = 0; i < nested_join->conditions.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << nested_join->conditions[i].left->ToString() << " "
                      << duckdb::ExpressionTypeToOperator(nested_join->conditions[i].comparison) << " "
                      << nested_join->conditions[i].right->ToString();
        }
        std::cout << std::endl;
    } else if (op->type == duckdb::PhysicalOperatorType::BLOCKWISE_NL_JOIN)
    {
        auto block_join = reinterpret_cast<duckdb::PhysicalBlockwiseNLJoin *>(op);

        // Print join conditions
        std::cout << indent << "Join Conditions: ";
        print_expression(block_join->condition.get(), indent.length() / 2 + 1);
        std::cout << std::endl;
    }
    
}

void projection(DuckDBManager &manager, duckdb::PhysicalOperator *op, std::string indent)
{
    auto projection = reinterpret_cast<duckdb::PhysicalProjection *>(op);
    std::cout << indent << "Projection Expressions: ";
    for (size_t i = 0; i < projection->select_list.size(); i++)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << projection->select_list[i]->ToString();
    }
    std::cout << std::endl;
}


void order_by(DuckDBManager &manager, duckdb::PhysicalOperator *op, std::string indent)
{
    auto order = reinterpret_cast<duckdb::PhysicalOrder *>(op);
    std::cout << indent << "Order By: ";
    for (size_t i = 0; i < order->orders.size(); i++)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << order->orders[i].expression->ToString() << " "
                  << (order->orders[i].type == duckdb::OrderType::ASCENDING ? "ASC" : "DESC");
    }
    std::cout << std::endl;
}


void traversePhysicalOperator(DuckDBManager &manager, duckdb::PhysicalOperator *op, int depth = 0)
{
    if (!op)
        return;

    // Print indentation based on depth
    std::string indent(depth * 2, ' ');
    for (auto &child : op->children)
        traversePhysicalOperator(manager, &child.get(), depth + 1);
    // Print information about the current operator
    std::cout << indent << "Operator Type: " << op->GetName() << std::endl;
    switch (op->type)
    {
    case duckdb::PhysicalOperatorType::TABLE_SCAN:
        table_scan(manager, op, indent);
        break;

    case duckdb::PhysicalOperatorType::FILTER:
        filter(manager, op, indent);
        break;

    case duckdb::PhysicalOperatorType::PROJECTION:
        projection(manager, op, indent);
        break;

    case duckdb::PhysicalOperatorType::BLOCKWISE_NL_JOIN:
    case duckdb::PhysicalOperatorType::HASH_JOIN:
    case duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN:
        nested_loop_join(manager, op, indent);
        break;

    case duckdb::PhysicalOperatorType::ORDER_BY:
        order_by(manager, op, indent);
        break;
    
    default:
        break;
    }
    std::cout << indent << "------------------------" << std::endl;
}

int main()
{
    try
    {
        auto db_manager = DuckDBManager::create();
        // Initialize tables from CSV files in a directory (schema only)
        std::string csv_directory = "./csv_data";
        db_manager.initializeTablesFromCSVs(csv_directory);
        db_manager.listAllTables();
        std::vector<std::string> test_queries = {
            //     // Simple select
                // "SELECT name FROM employees WHERE  salary > 5000  ",
                "SELECT e.name, e.salary, d.department_name " \
                "FROM employees e, departments d " \
                "WHERE e.salary > 50000 " \
                "AND d.department_name != 'HR' " \
                "AND (e.department_id = d.id and e.salary < d.id) or (e.id > d.id and e.salary > d.id)"  \
                "ORDER BY e.salary DESC"
            // "SELECT max(salary) "
            // "FROM employees "
            // "WHERE (salary > 50000 AND name='Brittany Gonzalez' AND hire_date >='2023-10-22') ",
            // "SELECT max(budget) "
            // "FROM projects ",
            // "ORDER BY salary DESC"
            // "SELECT name, salary, hire_date "
            // "FROM employees "
            // "WHERE (salary > 50000 AND name='Brittany Gonzalez' AND hire_date >='2023-10-22') "
            // "ORDER BY salary DESC"
            // "SELECT e.name, e.salary, d.department_name " \
            // "FROM employees e, departments d " \
            // "WHERE e.salary > 50000 " \
            // "AND d.department_name != 'HR' " \
            // "AND e.department_id = d.id " 
        };
        // std::cout << "=========================================" << std::endl;
        // // std::cout << "=========================================" << std::endl;
        // // std::cout << db_manager.readNextBatch("employees") << std::endl;
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
        std::cout << db_manager.readNextBatch("employees") << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}