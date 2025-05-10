#include "join.h"
#include "../headers/enums.h"
#include "../headers/constants.h"
#include <string.h>
template <typename T>
void addValueToVoidArray(void *array, size_t index, T value)
{
    T *typed_array = static_cast<T *>(array);
    typed_array[index] = value;
}

size_t addBatchColumns(void **result_table_batches, std::vector<bool> &matches, std::vector<std::shared_ptr<ColumnBatch>> &left_batch, std::vector<std::shared_ptr<ColumnBatch>> &right_batch)
{
    // now we have the memory for the result pointers
    size_t current_row = 0, left_size = left_batch[0]->getNumRows(), right_size = right_batch[0]->getNumRows();
    for (size_t i = 0; i < left_size; i++)
    {
        for (size_t j = 0; j < right_size; j++)
        {
            int index = i * right_size + j;
            if (matches[index])
            {
                for (size_t col_idx = 0; col_idx < left_batch.size(); col_idx++)
                {
                    switch (left_batch[col_idx]->getType())
                    {
                    case ColumnType::FLOAT:
                        addValueToVoidArray<float>(result_table_batches[col_idx], current_row, left_batch[col_idx]->getDouble(i));
                        break;
                    case ColumnType::STRING:
                    {
                        std::string str = left_batch[col_idx]->getString(i);
                        char *str_array = static_cast<char *>(result_table_batches[col_idx]);
                        strncpy(str_array + current_row * MAX_STRING_LENGTH, str.c_str(), MAX_STRING_LENGTH);
                        // Ensure null termination
                        str_array[current_row * MAX_STRING_LENGTH + std::min(str.length(), (size_t)MAX_STRING_LENGTH - 1)] = '\0';
                    }
                    break;
                    case ColumnType::DATE:
                        addValueToVoidArray<int64_t>(result_table_batches[col_idx], current_row, left_batch[col_idx]->getDateAsInt64(i));
                        break;
                    default:
                        throw "Invalid column type";
                    }
                }

                for (size_t col_idx = 0; col_idx < right_batch.size(); col_idx++)
                {
                    switch (right_batch[col_idx]->getType())
                    {
                    case ColumnType::FLOAT:
                        addValueToVoidArray<float>(result_table_batches[left_batch.size() + col_idx], current_row, right_batch[col_idx]->getDouble(j));
                        break;
                    case ColumnType::STRING:
                        // TODO: add string to result batch
                        {
                            std::string str = right_batch[col_idx]->getString(j);
                            char *str_array = static_cast<char *>(result_table_batches[left_batch.size() + col_idx]);
                            strncpy(str_array + current_row * MAX_STRING_LENGTH, str.c_str(), MAX_STRING_LENGTH);
                            // Ensure null termination
                            str_array[current_row * MAX_STRING_LENGTH + std::min(str.length(), (size_t)MAX_STRING_LENGTH - 1)] = '\0';
                        }
                        break;
                    case ColumnType::DATE:
                        addValueToVoidArray<int64_t>(result_table_batches[left_batch.size() + col_idx], current_row, right_batch[col_idx]->getDateAsInt64(j));
                        break;
                    default:
                        throw "Invalid column type";
                    }
                }
                current_row++;
            }
        }
    }
    return current_row;
}
std::string trimString(const std::string &str)
{
    // Find first non-whitespace
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    // Find last non-whitespace
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}
void **allocateResultTableBatches(const std::vector<bool> &matches,
                                  const std::vector<std::shared_ptr<ColumnBatch>> &left_batch,
                                  const std::vector<std::shared_ptr<ColumnBatch>> &right_batch)
{
    size_t num_of_matched = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        num_of_matched += matches[i];
    }

    void **result_table_batches = new void *[left_batch.size() + right_batch.size()];
    for (size_t i = 0; i < left_batch.size(); i++)
    {
        result_table_batches[i] = malloc(num_of_matched * sizeFromColumnType(left_batch[i]->getType()));
    }
    for (size_t i = 0; i < right_batch.size(); i++)
    {
        result_table_batches[left_batch.size() + i] = malloc(num_of_matched * sizeFromColumnType(right_batch[i]->getType()));
    }
    return result_table_batches;
}

bool compareJoinCondition(const JoinCondition &condition,
                          const std::vector<std::shared_ptr<ColumnBatch>> &left_batch,
                          const std::vector<std::shared_ptr<ColumnBatch>> &right_batch,
                          size_t left_row_idx, size_t right_row_idx)
{

    switch (condition.columnType)
    {
    case ColumnType::FLOAT:
        return compareValues<float>(
            left_batch[condition.leftColumnIdx]->getDouble(left_row_idx),
            right_batch[condition.rightColumnIdx]->getDouble(right_row_idx),
            condition.op);
    case ColumnType::STRING:
    {

        std::string left_untrimmed = left_batch[condition.leftColumnIdx]->getString(left_row_idx);
        std::string right_untrimmed = right_batch[condition.rightColumnIdx]->getString(right_row_idx);
        return compareValues<std::string>(
            trimString(left_untrimmed),
            trimString(right_untrimmed),
            condition.op);
    }
    case ColumnType::DATE:
        return compareValues<int64_t>(
            left_batch[condition.leftColumnIdx]->getDateAsInt64(left_row_idx),
            right_batch[condition.rightColumnIdx]->getDateAsInt64(right_row_idx),
            condition.op);
    default:
        return false;
    }
}

void joinTablesCPU(std::shared_ptr<Table> left_table, std::shared_ptr<Table> right_table,
                   std::vector<JoinCondition> join_conditions,
                   std::shared_ptr<Table> result_table)
{
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    int timeSum = 0;
    while (left_table->hasMoreData())
    {
        std::cout << "111111111111" << std::endl;
        left_table->readNextBatch();
        std::cout << "left table size: " << left_table->getCurrentBatchSize() << std::endl;
        std::vector<std::shared_ptr<ColumnBatch>> left_batches = left_table->getCurrentBatch();
        while (right_table->hasMoreData())
        {
            std::cout << "222222222222" << std::endl;
            right_table->readNextBatch();
            std::cout << "right table size: " << right_table->getCurrentBatchSize() << std::endl;
            std::vector<std::shared_ptr<ColumnBatch>> right_batches = right_table->getCurrentBatch();
            std::vector<bool> matches(left_batches[0]->getNumRows() * right_batches[0]->getNumRows(), false);
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < left_table->getCurrentBatchSize(); i++)
            {
                std::cout << "333333333333" << std::endl;
                for (size_t j = 0; j < right_table->getCurrentBatchSize(); j++)
                {
                    bool match = true;
                    for (size_t k = 0; k < join_conditions.size(); k++)
                    {
                        match = compareJoinCondition(join_conditions[k], left_batches, right_batches, i, j);
                        matches[i * right_batches[0]->getNumRows() + j] = match;
                    }
                    // This is after a single GPU batch
                    // now we have the bool array, we need to init void* for the result tables
                }
            }
            std::cout << "444444444444" << std::endl;
            void **result_table_batches = allocateResultTableBatches(matches, left_batches, right_batches);
            std::cout << "5555555555555" << std::endl;
            size_t num_rows = addBatchColumns(result_table_batches, matches, left_batches, right_batches);
            std::cout << "6666666666666" << std::endl;
            std::cout << "num rows matched: " << num_rows << std::endl;
            // now the void** should be the same as the GPU result that we get
            // we need to add the result to the result table
            result_table->addResultBatch(result_table_batches, num_rows);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            timeSum += duration.count();
        }
        right_table->resetFilePositionToStart();
    }
    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << "time taken by join duration: " << timeSum << " seconds" << std::endl;
}