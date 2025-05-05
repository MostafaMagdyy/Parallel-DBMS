#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <unordered_map>
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

    DuckDBManager();

public:
    static DuckDBManager create();
    ~DuckDBManager();

    DuckDBManager(const DuckDBManager &) = delete;
    DuckDBManager &operator=(const DuckDBManager &) = delete;

    DuckDBManager(DuckDBManager &&) = default;
    DuckDBManager &operator=(DuckDBManager &&) = default;

    std::shared_ptr<duckdb::Connection> getCon();
    static std::vector<ColumnMetadata> parseCSVHeader(const std::string &csv_file);
    static void createTableFromCSV(duckdb::DuckDB &db, duckdb::Connection &con, const std::string &csv_file);
    std::shared_ptr<Table> getTable(const std::string &table_name);
    void setBatchSize(size_t batch_size);
    void loadTableFromCSV(const std::string &csv_file, size_t batch_size = 0);
    void initializeTablesFromCSVs(const std::string &csv_directory, size_t batch_size = 0);
    bool readNextBatch(const std::string &table_name);
    duckdb::PhysicalOperator *getQueryPlan(const std::string &query);
    void listAllTables();
};
