#include "cpu_sort.h"

void sortTablesCPU(std::shared_ptr<Table> &table, std::string &columnName)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::shared_ptr<ColumnBatch>> column_batches = table->getCurrentBatch();
    size_t num_rows = column_batches[0]->getNumRows();
    size_t column_index = table->getColumnIndexProjected(columnName);
    ColumnType column_type = column_batches[column_index]->getType();
    std::vector<std::pair<std::variant<std::string, float, int64_t>, size_t>> column_data;
    column_data.reserve(num_rows);
    for (size_t i = 0; i < num_rows; i++)
    {
        switch (column_type)
        {
        case ColumnType::STRING:
            column_data.push_back({column_batches[column_index]->getString(i), i});
            break;
        case ColumnType::FLOAT:
            column_data.push_back({(column_batches[column_index]->getDouble(i)), i});
            break;
        case ColumnType::DATE:
            column_data.push_back({(column_batches[column_index]->getDateAsInt64(i)), i});
            break;
        default:
            throw std::runtime_error("Invalid column type");
        }
    }
    std::sort(column_data.begin(), column_data.end());
    std::vector<std::shared_ptr<ColumnBatch>> sorted_column_batches;
    for (size_t j = 0; j < column_batches.size(); j++)
    {
        std::shared_ptr<ColumnBatch> result_batch = std::make_shared<ColumnBatch>(column_batches[j]->getType(), num_rows);
        for (size_t i = 0; i < num_rows; i++)
        {
            size_t sorted_index = column_data[i].second;
            switch (column_batches[j]->getType())
            {
            case ColumnType::STRING:
                result_batch->addString(column_batches[j]->getString(sorted_index));
                break;
            case ColumnType::FLOAT:
                result_batch->addDouble(column_batches[j]->getDouble(sorted_index));
                break;
            case ColumnType::DATE:
                result_batch->addDate(column_batches[j]->getDateAsInt64(sorted_index));
                break;
            default:
                throw std::runtime_error("Invalid column type");
            }
        }
        sorted_column_batches.push_back(result_batch);
    }
    // Print sorted column batches
    std::cout << "Sorted Column Batches:" << std::endl;
    for (size_t j = 0; j < sorted_column_batches.size(); j++)
    {
        for (size_t i = 0; i < num_rows; i++)
        {
            switch (sorted_column_batches[j]->getType())
            {
            case ColumnType::STRING:
                break;
            case ColumnType::FLOAT:
                break;
            case ColumnType::DATE:
                break;
            default:
                std::cout << "Unknown type";
            }
        }
        std::cout << std::endl;
    }
    table->setCurrentBatch(sorted_column_batches);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << "Sorting took " << duration.count() << " seconds." << std::endl;
}