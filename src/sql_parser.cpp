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
#include <duckdb/execution/operator/order/physical_order.hpp>
#include <duckdb/execution/operator/order/physical_top_n.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <chrono>
#include <sys/resource.h>
#include "headers/column.h"
#include "headers/table.h"
namespace fs = std::filesystem;
class DuckDBManager
{
private:
    std::unique_ptr<duckdb::DuckDB> db;
    std::shared_ptr<duckdb::Connection> con;
    std::unordered_map<std::string, std::shared_ptr<Table>> tables;
    size_t default_batch_size;
    std::unique_ptr<duckdb::PhysicalPlan> physical_plan_local;

    // Private constructor to prevent direct instantiation without initialization
    DuckDBManager() : db(nullptr), con(nullptr) {}

public:
    static DuckDBManager create()
    {
        DuckDBManager manager;
        manager.db = std::make_unique<duckdb::DuckDB>(nullptr);
        manager.con = std::make_unique<duckdb::Connection>(*manager.db);
        manager.con->Query("BEGIN TRANSACTION");
        manager.con->Query("SET disabled_optimizers = 'statistics_propagation';");
        manager.default_batch_size = 1000000; // Default batch size
        return manager;
    }

    ~DuckDBManager()
    {
        con.reset();
        db.reset();
    }

    // Delete copy constructor and assignment operator to prevent copying
    DuckDBManager(const DuckDBManager &) = delete;
    DuckDBManager &operator=(const DuckDBManager &) = delete;

    // Move constructor and assignment operator for safe transfer
    DuckDBManager(DuckDBManager &&) = default;
    DuckDBManager &operator=(DuckDBManager &&) = default;

    std::shared_ptr<duckdb::Connection> getCon()
    {
        return con;
    }

    static std::vector<ColumnMetadata> parseCSVHeader(const std::string &csv_file)
    {
        std::ifstream file(csv_file);
        if (!file.is_open())
        {
            throw std::runtime_error("Unable to open CSV file: " + csv_file);
        }

        std::string header_line;
        std::getline(file, header_line);
        std::stringstream ss(header_line);
        std::string token;
        std::vector<ColumnMetadata> columns;
        size_t index = 0;

        while (std::getline(ss, token, ','))
        {
            size_t type_start = token.find('(');
            if (type_start == std::string::npos)
            {
                throw std::runtime_error("Invalid header format in CSV (missing type): " + token);
            }

            std::string col_name = token.substr(0, type_start);
            // Remove leading/trailing whitespace from column name
            col_name.erase(0, col_name.find_first_not_of(" \t"));
            col_name.erase(col_name.find_last_not_of(" \t") + 1);

            // Process all parenthesized parts
            std::string duckdb_type;
            bool is_primary_key = false;
            ColumnType col_type = ColumnType::UNKNOWN;

            size_t pos = 0;
            while ((pos = token.find('(', pos)) != std::string::npos)
            {
                size_t end_pos = token.find(')', pos);
                if (end_pos == std::string::npos)
                {
                    throw std::runtime_error("Invalid header format in CSV (unclosed parenthesis): " + token);
                }

                // Extract content inside parentheses
                std::string part = token.substr(pos + 1, end_pos - pos - 1);
                part.erase(0, part.find_first_not_of(" \t"));
                part.erase(part.find_last_not_of(" \t") + 1);

                if (part == "T")
                {
                    duckdb_type = "VARCHAR";
                    col_type = ColumnType::STRING;
                }
                else if (part == "N")
                {
                    duckdb_type = "FLOAT";
                    col_type = ColumnType::FLOAT;
                }
                else if (part == "D")
                {
                    duckdb_type = "TIMESTAMP";
                    col_type = ColumnType::DATE;
                }
                else if (part == "P")
                {
                    is_primary_key = true;
                }
                else
                {
                    throw std::runtime_error("Unsupported type or constraint in header: " + part);
                }

                pos = end_pos + 1;
            }

            // Make sure we have a type
            if (col_type == ColumnType::UNKNOWN)
            {
                throw std::runtime_error("No valid type specified for column: " + col_name);
            }

            // Use the correct constructor signature
            columns.push_back(ColumnMetadata(
                col_name,       // name
                col_type,       // type (enum)
                duckdb_type,    // duckdb_type
                is_primary_key, // is_primary_key
                index           // index
                ));

            index++;
        }

        file.close();
        return columns;
    }

    // Static function to create table in DuckDB from CSV
    static void createTableFromCSV(duckdb::DuckDB &db, duckdb::Connection &con, const std::string &csv_file)
    {
        std::string table_name = fs::path(csv_file).stem().string();
        auto columns = parseCSVHeader(csv_file);

        // Build CREATE TABLE statement
        std::stringstream create_sql;
        create_sql << "CREATE TABLE " << table_name << " (";
        for (size_t i = 0; i < columns.size(); ++i)
        {
            if (i > 0)
                create_sql << ", ";
            create_sql << columns[i].name << " " << columns[i].duckdb_type;
            if (columns[i].is_primary_key)
            {
                create_sql << " PRIMARY KEY";
            }
        }
        create_sql << ");";

        // Execute CREATE TABLE
        con.Query(create_sql.str());
        std::cout << "Query is " << create_sql.str() << std::endl;
        std::cout << "Created table: " << table_name << " (schema only) from " << csv_file << std::endl;
    }

    std::shared_ptr<Table> getTable(const std::string &table_name)
    {
        auto it = tables.find(table_name);
        if (it != tables.end())
        {
            return it->second;
        }
        return nullptr;
    }
    // Add table management functions
    void setBatchSize(size_t batch_size)
    {
        default_batch_size = batch_size;
    }

    // Load tables from CSV
    void loadTableFromCSV(const std::string &csv_file, size_t batch_size = 0)
    {
        if (!db || !con)

            throw std::runtime_error("DuckDB not initialized.");

        if (batch_size == 0)

            batch_size = default_batch_size;

        std::string table_name = fs::path(csv_file).stem().string();
        std::cout << table_name << '\n';
        auto columns = parseCSVHeader(csv_file);
        createTableFromCSV(*db, *con, csv_file);
        auto table = std::make_shared<Table>(table_name, columns, csv_file, batch_size);
        tables[table_name] = table;
    }
    void initializeTablesFromCSVs(const std::string &csv_directory, size_t batch_size = 0)
    {
        if (!db || !con)
        {
            throw std::runtime_error("DuckDB not initialized.");
        }

        for (const auto &entry : fs::directory_iterator(csv_directory))
        {
            if (entry.path().extension() == ".csv")
            {
                loadTableFromCSV(entry.path().string(), batch_size);
            }
        }
    }
    bool readNextBatch(const std::string &table_name)
    {
        auto it = tables.find(table_name);
        if (it != tables.end())
        {
            bool res = it->second->readNextBatch();
            if (res)
                it->second->printCurrentBatch();
            return res;
        }
        return false;
    }
    duckdb::PhysicalOperator *getQueryPlan(const std::string &query)
    {
        try
        {
            duckdb::Parser parser;
            parser.ParseQuery(query);

            if (parser.statements.empty())
            {
                throw std::runtime_error("No valid SQL statement found in query");
            }
            auto statements = std::move(parser.statements);
            duckdb::Planner planner(*con->context);
            planner.CreatePlan(std::move(statements[0]));

            duckdb::Optimizer optimizer(*planner.binder, *con->context);
            auto logical_plan = optimizer.Optimize(std::move(planner.plan));
            duckdb::PhysicalPlanGenerator physical_plan_generator(*con->context);
            physical_plan_local = physical_plan_generator.Plan(logical_plan->Copy(*con->context));
            physical_plan_local->Root().Print();
            return &physical_plan_local->Root();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error generating query plan: " << e.what() << std::endl;
            throw;
        }
    }
    void listAllTables()
    {
        std::cout << "Tables in DuckDB:" << std::endl;
        for (const auto &pair : tables)
        {
            std::cout << " - " << pair.first << std::endl;
        }
    }
};

void traversePhysicalOperator(DuckDBManager &manager, duckdb::PhysicalOperator *op, int depth = 0)
{
    if (!op)
        return;

    // Print indentation based on depth
    std::string indent(depth * 2, ' ');

    // Print information about the current operator
    std::cout << indent << "Operator Type: " << op->GetName() << std::endl;

    // Print operator-specific information based on operator type
    switch (op->type)
    {
    case duckdb::PhysicalOperatorType::TABLE_SCAN:
    {
        auto scan = reinterpret_cast<duckdb::PhysicalTableScan *>(op);
        auto params = scan->ParamsToString();
        if (params.find("Table") != params.end())
        {
            std::cout << indent << "Table Name: " << params["Table"] << std::endl;
        }
        // Show filters if available
        if (params.find("Table") != params.end() && scan->table_filters && !scan->table_filters->filters.empty())
        {
            auto table = manager.getTable(params["Table"]);
            table->clearFilters();
            std::vector<FilterCondition> filter_conditions;
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

        if (!params.empty())
        {
            std::cout << indent << "Parameters:" << std::endl;
            for (const auto &pair : params)
            {
                std::cout << indent << "  " << pair.first << ": " << pair.second << std::endl;
            }
        }

        break;
    }

    case duckdb::PhysicalOperatorType::FILTER:
    {
        auto filter = reinterpret_cast<duckdb::PhysicalFilter *>(op);
        std::cout << indent << "Filter Expressions: ";
        if (filter->expression)
            std::cout << filter->expression->ToString() << std::endl;
        break;
    }

    case duckdb::PhysicalOperatorType::PROJECTION:
    {
        auto projection = reinterpret_cast<duckdb::PhysicalProjection *>(op);
        std::cout << indent << "Expressions: ";
        for (size_t i = 0; i < projection->select_list.size(); i++)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << projection->select_list[i]->ToString();
        }
        std::cout << std::endl;
        break;
    }

    case duckdb::PhysicalOperatorType::HASH_JOIN:
    case duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN:
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
        }
        break;
    }

    case duckdb::PhysicalOperatorType::ORDER_BY:
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
        break;
    }
        /*
        We need to add more physical operators here
        */

    default:
        break;
    }
    std::cout << indent << "------------------------" << std::endl;
    for (auto &child : op->children)
        traversePhysicalOperator(manager, &child.get(), depth + 1);
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
            //     // "SELECT name FROM users WHERE age > 25 AND dept_id = 1",
            //     "SELECT e.name, e.salary, d.department_name " \
            //     "FROM employees e, departments d " \
            //     "WHERE e.salary > 50000 " \
            //     "AND d.department_name != 'HR' " \
            //     "AND e.department_id = d.id " \
            //     "ORDER BY e.salary DESC"
            "SELECT name, hire_date "
            "FROM employees "
            "WHERE (salary > 50000 AND name='Brittany Gonzalez')",
            // "ORDER BY salary DESC"
            // "SELECT name, salary, hire_date "
            // "FROM employees "
            // "WHERE (salary > 50000 AND name='Brittany Gonzalez' AND hire_date >='2023-10-22') "
            // "ORDER BY salary DESC"
        };
        // "SELECT project_id,project_name,start_date " \
                // "FROM projects WHERE project_id > 5465 AND start_date> '2020-10-16' AND budget > 500 " \
            // };
        // // std::cout << "=========================================" << std::endl;
        // // std::cout << db_manager.readNextBatch("employees") << std::endl;
        // // std::cout << "=========================================" << std::endl;
        // // std::cout << db_manager.readNextBatch("employees") << std::endl;
        // // std::cout << "=========================================" << std::endl;
        // // std::cout << db_manager.readNextBatch("employees") << std::endl;
        // // std::cout << "=========================================" << std::endl;
        // // std::cout << db_manager.readNextBatch("employees") << std::endl;
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