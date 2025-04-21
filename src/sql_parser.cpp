#include <iostream>
#include <vector>
#include <duckdb.hpp>
#include <string>
#include <sstream>
#include <memory>
#include <unordered_set>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_join.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
using namespace duckdb;

// Function to traverse get table names from logical operators
void getTableNames(LogicalOperator *op, std::vector<std::string> &table_names)
{
    if (!op)
        return;

    if (op->type == LogicalOperatorType::LOGICAL_GET)
    {
        auto get = reinterpret_cast<LogicalGet *>(op);
        if (auto table = get->GetTable())
        {
            table_names.push_back(table->name);
        }
        else if (!get->function.name.empty())
        {
            table_names.push_back(get->function.name);
        }
    }

    // Recursively check children
    for (auto &child : op->children)
    {
        getTableNames(child.get(), table_names);
    }
}
// Function to traverse the logical operator tree
void traverseLogicalOperator(LogicalOperator *op, int depth = 0)
{
    if (!op)
        return;

    // Print indentation based on depth
    std::string indent(depth * 2, ' ');

    // Print information about the current operator
    std::cout << indent << "Operator Type: " << op->GetName() << std::endl;

    // Print operator-specific information
    switch (op->type)
    {
    case LogicalOperatorType::LOGICAL_PROJECTION:
    {
        auto projection = reinterpret_cast<LogicalProjection *>(op);
        std::cout << indent << "Expressions: ";
        for (size_t i = 0; i < projection->expressions.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << projection->expressions[i]->ToString();
        }
        std::cout << std::endl;
        break;
    }
    case LogicalOperatorType::LOGICAL_FILTER:
    {
        auto filter = reinterpret_cast<LogicalFilter *>(op);
        std::cout << indent << "Filter Expressions: ";
        for (size_t i = 0; i < filter->expressions.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << filter->expressions[i]->ToString();
        }
        std::cout << std::endl;
        break;
    }
    case LogicalOperatorType::LOGICAL_GET:
    {
        auto get = reinterpret_cast<LogicalGet *>(op);
        // Use GetTable() to get table information if available
        if (auto table = get->GetTable())
        {
            std::cout << indent << "Table: " << table->name << std::endl;
        }
        else
        {
            std::cout << indent << "Table: " << "(Function Scan)" << std::endl;
        }
        // Print returned column names
        if (!get->table_filters.filters.empty())
        {
            std::cout << indent << "Pushed-down Filters:" << std::endl;
            for (auto &kv : get->table_filters.filters)
            {
                auto &column_index = kv.first;
                auto &filter = kv.second;
                // Get the column name if available
                string column_name;
                if (column_index < get->names.size())
                {
                    column_name = get->names[column_index];
                }
                else
                {
                    column_name = "col_" + std::to_string(column_index);
                }
                // Use the filter's ToString method with the column name
                std::cout << indent << "  " << filter->ToString(column_name) << std::endl;
            }
        }
        std::cout << std::endl;
        break;
    }
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    {
        auto join = reinterpret_cast<LogicalComparisonJoin *>(op);
        std::cout << indent << "Join Type: " << JoinTypeToString(join->join_type) << std::endl;

        // Print join conditions if available
        if (!join->conditions.empty())
        {
            std::cout << indent << "Join Conditions: ";
            for (size_t i = 0; i < join->conditions.size(); i++)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << join->conditions[i].left->ToString() << " "
                          << ExpressionTypeToOperator(join->conditions[i].comparison) << " "
                          << join->conditions[i].right->ToString();
            }
            std::cout << std::endl;
        }
        else if (!join->expressions.empty())
        {
            // Some joins might still use expressions for conditions
            std::cout << indent << "Join Expressions: ";
            for (size_t i = 0; i < join->expressions.size(); i++)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << join->expressions[i]->ToString();
            }
            std::cout << std::endl;
        }

        std::unordered_set<idx_t> left_bindings;
        std::unordered_set<idx_t> right_bindings;

        if (!join->children.empty() && join->children.size() >= 2)
        {

            // Additionally, get the actual table names
            std::vector<std::string> left_tables;
            std::vector<std::string> right_tables;

            getTableNames(join->children[0].get(), left_tables);
            getTableNames(join->children[1].get(), right_tables);

            std::cout << indent << "Left Tables: ";
            for (size_t i = 0; i < left_tables.size(); i++)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << left_tables[i];
            }
            std::cout << std::endl;

            std::cout << indent << "Right Tables: ";
            for (size_t i = 0; i < right_tables.size(); i++)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << right_tables[i];
            }
            std::cout << std::endl;
        }

        // Print predicate if available (your existing code)
        if (join->predicate)
        {
            std::cout << indent << "Additional Predicate: " << join->predicate->ToString() << std::endl;
        }

        break;
    }
    case LogicalOperatorType::LOGICAL_ORDER_BY:
    {
        auto order = reinterpret_cast<LogicalOrder *>(op);
        std::cout << indent << "Order By: ";
        for (size_t i = 0; i < order->orders.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << order->orders[i].expression->ToString() << " "
                      << (order->orders[i].type == OrderType::ASCENDING ? "ASC" : "DESC");
        }
        std::cout << std::endl;
        break;
    }
    default:
        break;
    }

    std::cout << indent << "------------------------" << std::endl;

    // Recursively traverse child nodes
    for (auto &child : op->children)
    {
        traverseLogicalOperator(child.get(), depth + 1);
    }
}

int main()
{
    try
    {
        DuckDB db(nullptr);
        Connection con(db);
        con.Query("SET disabled_optimizers = 'statistics_propagation';");
        con.Query("CREATE TABLE users(id INTEGER, name VARCHAR, age INTEGER, dept_id INTEGER);");
        con.Query(R"(
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                department_name TEXT NOT NULL
            );
        )");
        con.Query(R"(CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            salary DOUBLE,
            department_id INTEGER,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );)");

        // Test different query types
        std::vector<std::string> test_queries = {
            // Simple select
            // "SELECT name FROM users WHERE age > 25 AND dept_id = 1",

            // // Join query
            // "SELECT employees.name, departments.department_name FROM employees "
            // "INNER JOIN departments ON employees.department_id > departments.id",

            // // Query with ORDER BY
            // "SELECT name, salary FROM employees ORDER BY salary DESC",

            // // Complex query with multiple conditions
            "SELECT e.name, e.salary, d.department_name "
            "FROM employees e "
            "LEFT JOIN departments d ON e.department_id = d.id "
            "WHERE e.salary > 50000 AND d.department_name != 'HR' "
            "ORDER BY e.salary DESC"

            // "SELECT e.name, e.salary, d.department_name "
            // "FROM employees e "
            // "JOIN departments d ON e.department_id = d.id "
            // "WHERE e.salary > ("
            // "    SELECT e2.salary "
            // "    FROM employees e2 "
            // "    WHERE e2.department_id = e.department_id "
            // ") "
            // "ORDER BY e.salary DESC"
        };

        for (auto &query : test_queries)
        {
            std::cout << "\n=========================================" << std::endl;
            std::cout << "Planning query: " << query << std::endl;
            std::cout << "=========================================\n"
                      << std::endl;

            // Get the logical plan
            auto plan = con.ExtractPlan(query);

            // Print the default tree visualization
            std::cout << "Default plan visualization:" << std::endl;
            plan->Print();
            std::cout << std::endl;

            // Use our custom traversal function
            std::cout << "Custom tree traversal:" << std::endl;
            traverseLogicalOperator(plan.get());
            std::cout << std::endl;
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}