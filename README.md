# Parallel-DBMS: GPU-Accelerated Query Processing Database System

## 📘 Overview

**Parallel-DBMS** is a high-performance database system that integrates the [DuckDB](https://duckdb.org/) engine for SQL parsing and logical query planning, while offloading compute-intensive operations—such as **aggregations**, **sorting**, and **joins**—to the **GPU**. This hybrid model merges the expressive power of SQL with the massive parallelism of GPUs, enabling efficient analytical query execution on large-scale datasets.

---
## 🧠 System Design

### Query Planning:
DuckDB parses SQL queries and generates a logical query plan tree. The system then traverses this tree to identify and execute operations such as filters, projections, joins, and aggregations.

### Pushdown Optimization:
Filters and projections are pushed down to the scan level, ensuring only necessary columns and rows are read from disk, significantly reducing memory usage and I/O overhead.

### GPU Offloading:
- Operations like **aggregations**, **sorting**, and **joins** are accelerated on the GPU for optimal throughput
- String columns are handled on the CPU, while numeric and date fields benefit from GPU acceleration

### Batch Processing:
Data is processed in user-defined batches. Only the filtered and projected columns are loaded into memory and passed to the GPU.

---

## ⚙️ GPU Kernel Design

### 🔸 Aggregates
- **Warp-Level Reductions** for fast intra-warp operations using warp shuffles
- **Block-Level Reductions** via `atomicAdd`, `atomicMax`, and `atomicMin` ensure correctness with parallel updates

### 🔸 Sorting
**4-Way Radix Sort**:
1. Local sort within thread blocks
2. Global offset calculation using prefix sums
3. Final merge across blocks to produce the global sort

### 🔸 Join
**Block-Nested Loop Join**:
- Smaller table fits into shared memory
- Parallel probing of larger table by threads to perform the join efficiently

---

## 📊 Performance Analysis

### Aggregates

| Data Size | Rows  | Batch Size | Type  | Function | CUDA memcpy | Kernel  | CUDA malloc | Total CUDA | CPU only | Query Total Time |
|-----------|-------|------------|-------|----------|-------------|---------|-------------|------------|----------|------------------|
| ~0.5GB    | 10M   | 1M         | Float | AVG      | 13.8ms      | 5.9ms   | 22ms        | 41.7ms     | 96ms     | 12.2s            |
| ~0.5GB    | 10M   | 1M         | Date  | MAX      | 14.2ms      | 0.5ms   | 16ms        | 30.7ms     | 89ms     | 28s              |
| ~900MB    | 20M   | 10M        | Float | AVG      | 25.8ms      | 12ms    | 49ms        | 86.8ms     | 111ms    | 23.3s            |
| ~900MB    | 20M   | 10M        | Date  | MAX      | 39.4ms      | 1.1ms   | 40ms        | 81ms       | 166ms    | 57s              |

### Sorting

| Data Size | Rows  | Type   | CUDA memcpy | Kernel  | CUDA malloc | Total CUDA | CPU  | Query Total Time |
|-----------|-------|--------|-------------|---------|-------------|------------|------|------------------|
| ~500MB   | 10M   | Float  | 539ms       | 4412ms  | 56ms        | 5007ms     | 42s  | 300s             |
| ~500MB   | 10M   | Date   | 531ms       | 4413ms  | 62ms        | 5006ms     | 47s  | 306s             |
| ~500MB   | 10M   | String | —           | —       | —           | —          | 50s  | —                |

---

## ⛔️ Query Language Support & Limitations

Supported SQL-like operations:
- `SELECT`, `FROM`, `WHERE`, `ORDER BY`, `JOIN` (via `WHERE` only)
- Filtering with **relational operators**: `<`, `>`, `=`
- Use of **logical operators**: `AND`, `OR`
- Aggregate functions: `COUNT`, `MAX`, `MIN`, `SUM`, `AVG`
- `ORDER BY` supports **ascending (Asc)** and **descending (Desc)** orders

### ✅ Valid Query Examples

```sql
SELECT s.Name, a.City, COUNT(*) AS total_count
FROM Students s, Addresses a
WHERE s.student_id = a.student_id AND s.year > 3
ORDER BY s.Name Asc;
```

---

## 🛠️ Installation & Setup

### Step 1: Install DuckDB

Clone and build DuckDB from its [official repository](https://github.com/duckdb/duckdb):

```bash
git clone https://github.com/duckdb/duckdb.git
cd duckdb
make
```

Ensure the compiled DuckDB library is placed in the correct path within the `Parallel-DBMS\src` directory structure.

### Step 2: Clone Parallel-DBMS

```bash
git clone https://github.com/MostafaMagdyy/Parallel-DBMS
cd Parallel-DBMS
```

### Step 3: Compile The Project

```bash
./run.sh
```

---

## � Usage

### Run the system

```bash
./sql_dbms <csv_directory> "<SQL query>"
```

- `<csv_directory>`: Path to a folder containing the input CSV files
- `"<SQL query>"`: A SQL-like query string. **Use double quotes** to wrap the query

#### Example

```bash
./sql_dbms ./data "SELECT Name, MAX(Salary) FROM employees WHERE Salary > 70000 ORDER BY Name Asc"
```

---

## 📁 CSV File Format

Each CSV file should begin with a **typed header**, specifying both data types and primary keys:

```csv
Employees_id (N) (P),Name (T),Salary (N),JoinedDate (D)
1,Alice Johnson,70000.5,2023-01-15 08:45:00
2,Robert Smith,65800.75,2022-08-05 09:30:00
3,Emily Davis,80000,2021-11-20 03:15:00
4,Michael Brown,72000.25,2023-11-03 12:00:00
5,Jessica Williams,75000.9,2020-02-25 10:50:00
```

- `(N)` = Numeric
- `(T)` = Text (string)
- `(D)` = Date (`YYYY-MM-DD HH:MM:SS` format)
- `(P)` = Primary Key

---