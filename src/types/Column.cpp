#include<string>
enum class ColumnType {
    STRING,    // VARCHAR/TEXT
    DOUBLE,    // NUMERIC/FLOAT
    DATE,      // TIMESTAMP
    UNKNOWN
};

// Helper function to convert ColumnType to string
std::string columnTypeToString(ColumnType type) {
    switch(type) {
        case ColumnType::STRING: return "STRING";
        case ColumnType::DOUBLE: return "DOUBLE";
        case ColumnType::DATE: return "DATE";
        default: return "UNKNOWN";
    }
}

// Helper function to convert string type to ColumnType
ColumnType stringToColumnType(const std::string& type_str) {
    if (type_str == "VARCHAR") return ColumnType::STRING;
    if (type_str == "DOUBLE") return ColumnType::DOUBLE;
    if (type_str == "TIMESTAMP") return ColumnType::DATE;
    return ColumnType::UNKNOWN;
}

// Replace the existing ColumnMetadata struct
struct ColumnMetadata {
    std::string name;         // Column name
    ColumnType type;          // Column type (enum)
    std::string duckdb_type;  // Original DuckDB type name
    bool is_primary_key;      // Flag to indicate if column is a primary key
    size_t index;             // Column index in table
    size_t byte_offset;       // Byte offset in the row structure
    size_t element_size;      // Size in bytes for the element (for fixed-size types)
    
    // Constructor
    ColumnMetadata(const std::string& name, ColumnType type, const std::string& duckdb_type,
                 bool is_primary_key, size_t index, size_t byte_offset = 0, size_t element_size = 0)
        : name(name), type(type), duckdb_type(duckdb_type), is_primary_key(is_primary_key),
          index(index), byte_offset(byte_offset), element_size(element_size) {}
};