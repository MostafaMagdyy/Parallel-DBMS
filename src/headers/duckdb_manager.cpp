#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <memory>
#include "duckdb_manager.h"
// --------duckdb includes----------------
#include <duckdb.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/execution/physical_operator.hpp>
#include <duckdb/execution/operator/order/physical_order.hpp>
#include <duckdb/execution/operator/order/physical_top_n.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <chrono>
#include "headers/column.h"
#include "headers/table.h"
namespace fs = std::filesystem;
std::unique_ptr<duckdb::DuckDB> db;
std::shared_ptr<duckdb::Connection> con;
std::unordered_map<std::string, std::shared_ptr<Table>> tables;
size_t default_batch_size;
std::unique_ptr<duckdb::PhysicalPlan> physical_plan_local;

DuckDBManager::DuckDBManager() : db(nullptr), con(nullptr) {}

DuckDBManager DuckDBManager::create()
{
    DuckDBManager manager;
    manager.db = std::make_unique<duckdb::DuckDB>(nullptr);
    manager.con = std::make_unique<duckdb::Connection>(*manager.db);
    manager.con->Query("BEGIN TRANSACTION");
    manager.con->Query("SET disabled_optimizers = 'statistics_propagation';");
    manager.default_batch_size = 1e7; // Default batch size
    return manager;
}

DuckDBManager::~DuckDBManager()
{
    con.reset();
    db.reset();
}

std::shared_ptr<duckdb::Connection> DuckDBManager::getCon()
{
    return con;
}

std::vector<ColumnMetadata> DuckDBManager::parseCSVHeader(const std::string &csv_file)
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
        col_name.erase(0, col_name.find_first_not_of(" \t"));
        col_name.erase(col_name.find_last_not_of(" \t") + 1);

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
                duckdb_type = "DOUBLE";
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

        if (col_type == ColumnType::UNKNOWN)
        {
            throw std::runtime_error("No valid type specified for column: " + col_name);
        }

        columns.push_back(ColumnMetadata(
            col_name,
            col_type,
            duckdb_type,
            is_primary_key,
            index));

        index++;
    }

    file.close();
    return columns;
}

void DuckDBManager::createTableFromCSV(duckdb::DuckDB &db, duckdb::Connection &con, const std::string &csv_file)
{
    std::string table_name = fs::path(csv_file).stem().string();
    auto columns = parseCSVHeader(csv_file);
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

    con.Query(create_sql.str());
    std::cout << "Query is " << create_sql.str() << std::endl;
    std::cout << "Created table: " << table_name << " (schema only) from " << csv_file << std::endl;
}

std::shared_ptr<Table> DuckDBManager::getTable(const std::string &table_name)
{
    auto it = tables.find(table_name);
    if (it != tables.end())
    {
        return it->second;
    }
    return nullptr;
}

void DuckDBManager::setBatchSize(size_t batch_size)
{
    default_batch_size = batch_size;
}

void DuckDBManager::loadTableFromCSV(const std::string &csv_file, size_t batch_size)
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

void DuckDBManager::initializeTablesFromCSVs(const std::string &csv_directory, size_t batch_size)
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

bool DuckDBManager::readNextBatch(const std::string &table_name)
{
    auto it = tables.find(table_name);
    if (it != tables.end())
    {
        bool res = it->second->readNextBatch();
        return res;
    }
    return false;
}

void DuckDBManager::printCurrentBatch(const std::string &table_name, size_t max_rows, size_t max_string_length)
{
    auto it = tables.find(table_name);
    if (it != tables.end())
    {
        it->second->printCurrentBatch(max_rows, max_string_length);
    }
    else
    {
        std::cerr << "Table not found: " << table_name << std::endl;
    }
}

duckdb::PhysicalOperator *DuckDBManager::getQueryPlan(const std::string &query)
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

void DuckDBManager::addTable(std::shared_ptr<Table> table)
{
    tables[table->getName()] = table;
}

std::string DuckDBManager::createTempTableName(const std::string &base_name)
{
    std::string temp_name = base_name + "_temp_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    return temp_name;
}

void DuckDBManager::listAllTables()
{
    std::cout << "Tables in DuckDB:" << std::endl;
    for (const auto &pair : tables)
    {
        std::cout << " - " << pair.first << std::endl;
    }
}
