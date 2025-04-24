#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <duckdb.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_join.hpp>
#include <duckdb/planner/operator/logical_order.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
using namespace duckdb;
namespace fs = std::filesystem;
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


// Structure to hold column metadata
struct ColumnMetadata {
    std::string name;          // Column name
    std::string duckdb_type;   // DuckDB type (e.g., VARCHAR, TIMESTAMP, DOUBLE)
    bool is_primary_key;       // Flag to indicate if column is a primary key
    size_t index;              // Column index in table
};

// Class to manage DuckDB operations
class DuckDBManager {
private:
    std::unique_ptr<DuckDB> db;      
    std::unique_ptr<Connection> con;  

    // Private constructor to prevent direct instantiation without initialization
    DuckDBManager() : db(nullptr), con(nullptr) {}

public:
    static DuckDBManager create() {
        DuckDBManager manager;
        manager.db = std::make_unique<DuckDB>(nullptr); 
        manager.con = std::make_unique<Connection>(*manager.db);
        manager.con->Query("SET disabled_optimizers = 'statistics_propagation, filter_pushdown';");
        return manager;
    }

    ~DuckDBManager() {
        con.reset(); 
        db.reset();  
    }

    // Delete copy constructor and assignment operator to prevent copying
    DuckDBManager(const DuckDBManager&) = delete;
    DuckDBManager& operator=(const DuckDBManager&) = delete;

    // Move constructor and assignment operator for safe transfer
    DuckDBManager(DuckDBManager&&) = default;
    DuckDBManager& operator=(DuckDBManager&&) = default;

    static std::vector<ColumnMetadata> parseCSVHeader(const std::string& csv_file) {
        std::ifstream file(csv_file);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open CSV file: " + csv_file);
        }
    
        std::string header_line;
        std::getline(file, header_line);
        std::stringstream ss(header_line);
        std::string token;
        std::vector<ColumnMetadata> columns;
        size_t index = 0;
    
        while (std::getline(ss, token, ',')) {
            size_t type_start = token.find('(');
            if (type_start == std::string::npos) {
                throw std::runtime_error("Invalid header format in CSV (missing type): " + token);
            }
    
            std::string col_name = token.substr(0, type_start);
            // Remove leading/trailing whitespace from column name
            col_name.erase(0, col_name.find_first_not_of(" \t"));
            col_name.erase(col_name.find_last_not_of(" \t") + 1);
    
            // Process all parenthesized parts
            std::string duckdb_type;
            bool is_primary_key = false;
            
            size_t pos = 0;
            while ((pos = token.find('(', pos)) != std::string::npos) {
                size_t end_pos = token.find(')', pos);
                if (end_pos == std::string::npos) {
                    throw std::runtime_error("Invalid header format in CSV (unclosed parenthesis): " + token);
                }
                
                // Extract content inside parentheses
                std::string part = token.substr(pos + 1, end_pos - pos - 1);
                part.erase(0, part.find_first_not_of(" \t"));
                part.erase(part.find_last_not_of(" \t") + 1);
                
                // Determine what this part represents
                if (part == "T") {
                    duckdb_type = "VARCHAR";
                } else if (part == "N") {
                    duckdb_type = "DOUBLE";
                } else if (part == "D") {
                    duckdb_type = "TIMESTAMP";
                } else if (part == "P") {
                    is_primary_key = true;
                } else {
                    throw std::runtime_error("Unsupported type or constraint in header: " + part);
                }
                
                pos = end_pos + 1;
            }
            
            // Make sure we have a type
            if (duckdb_type.empty()) {
                throw std::runtime_error("No valid type specified for column: " + col_name);
            }
    
            columns.push_back({col_name, duckdb_type, is_primary_key, index++});
        }
    
        file.close();
        return columns;
    }

    // Static function to create table in DuckDB from CSV
    static void createTableFromCSV(DuckDB& db, Connection& con, const std::string& csv_file) {
        std::string table_name = fs::path(csv_file).stem().string();
        auto columns = parseCSVHeader(csv_file);

        // Build CREATE TABLE statement
        std::stringstream create_sql;
        create_sql << "CREATE TABLE " << table_name << " (";
        for (size_t i = 0; i < columns.size(); ++i) {
            if (i > 0) create_sql << ", ";
            create_sql << columns[i].name << " " << columns[i].duckdb_type;
            if (columns[i].is_primary_key) {
                create_sql << " PRIMARY KEY";
            }
        }
        create_sql << ");";

        // Execute CREATE TABLE
        con.Query(create_sql.str());
        std::cout<<"Query is "<<create_sql.str()<<std::endl;
        std::cout << "Created table: " << table_name << " (schema only) from " << csv_file << std::endl;
    }

    // Function to initialize tables from a directory of CSV files (schema only)
    void initializeTablesFromCSVs(const std::string& csv_directory) {
        if (!db || !con) {
            throw std::runtime_error("DuckDB not initialized.");
        }
        for (const auto& entry : fs::directory_iterator(csv_directory)) {
            if (entry.path().extension() == ".csv") {
                createTableFromCSV(*db, *con, entry.path().string());
            }
        }
    }

    void listTables() {
        if (!con) {
            throw std::runtime_error("DuckDB connection not initialized.");
        }
        auto result = con->Query("SHOW TABLES;");
        std::cout << "\nCreated Tables in DuckDB:\n";
        for (size_t i = 0; i < result->RowCount(); ++i) {
            std::cout << result->GetValue(0, i).ToString() << std::endl;
        }
    }

    std::unique_ptr<LogicalOperator> getQueryPlan(const std::string& query) {
        if (!con) {
            throw std::runtime_error("DuckDB connection not initialized.");
        }
        std::cout << "Generating query plan for: " << query << std::endl;
        return con->ExtractPlan(query);
    }
};

int main()
{
    try
    {
        DuckDBManager db_manager = DuckDBManager::create();

        // Initialize tables from CSV files in a directory (schema only)
        std::string csv_directory = "./csv_data";
        db_manager.initializeTablesFromCSVs(csv_directory);
        // db_manager.listTables();
        std::vector<std::string> test_queries = {
        //     // Simple select
        //     // "SELECT name FROM users WHERE age > 25 AND dept_id = 1",
        //     "SELECT e.name, e.salary, d.department_name " \
        //     "FROM employees e, departments d " \
        //     "WHERE e.salary > 50000 " \
        //     "AND d.department_name != 'HR' " \
        //     "AND e.department_id = d.id " \
        //     "ORDER BY e.salary DESC"
        "SELECT name, salary, hire_date "\
        "FROM employees "\
        "WHERE salary > 50000 "\
        "ORDER BY salary DESC"
        };

        for (auto &query : test_queries)
        {
            std::cout << "\n=========================================" << std::endl;
            std::cout << "Planning query: " << query << std::endl;
            std::cout << "=========================================\n"
                      << std::endl;

            // Get the logical plan
            auto plan = db_manager.getQueryPlan(query);
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