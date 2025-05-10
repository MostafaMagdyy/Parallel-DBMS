#include "cpu_sort.h"
void sortTablesCPU(std::shared_ptr<Table> &table, std::string &columnName)
{
    std::vector<std::shared_ptr<ColumnBatch>> column_batches = table->getCurrentBatch();
    size_t num_rows = column_batches[0]->getNumRows();
    size_t column_index = table->getColumnIndexProjected(columnName);

    // Create indexes and sort them based on the column values
    std::vector<std::pair<std::string, size_t>> column_data;
    for (size_t i = 0; i < num_rows; i++)
    {
        column_data.push_back({column_batches[column_index]->getString(i), i});
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
    for (size_t j = 0; j < sorted_column_batches.size(); j++) {
        std::cout << "Column " << j << " (Type: " << static_cast<int>(sorted_column_batches[j]->getType()) << "):" << std::endl;
        for (size_t i = 0; i < num_rows; i++) {
            std::cout << "  Row " << i << ": ";
            switch (sorted_column_batches[j]->getType()) {
                case ColumnType::STRING:
                    std::cout << sorted_column_batches[j]->getString(i);
                    break;
                case ColumnType::FLOAT:
                    std::cout << sorted_column_batches[j]->getDouble(i);
                    break;
                case ColumnType::DATE:
                    std::cout << sorted_column_batches[j]->getDateAsInt64(i);
                    break;
                default:
                    std::cout << "Unknown type";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    table->setCurrentBatch(sorted_column_batches);
    table->printCurrentBatch(10, 30);
}