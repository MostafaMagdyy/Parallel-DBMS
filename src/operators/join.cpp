#include "join.h"
#include "../headers/enums.h"
#include "cuda_runtime.h"
#include "../headers/table.h"
#include "../headers/device_struct.h"
#include "../Join/join.h"
#include "../Join/globals.cuh"
#include "../headers/column.h"

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
                        // TODO: add string to result batch
                        throw std::runtime_error("String column not supported in result batch");
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
                        throw std::runtime_error("String column not supported in result batch");
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

    for (size_t i = 0; i < left_batch.size() + right_batch.size(); i++)
    {
        free(result_table_batches[i]);
    }
    free(result_table_batches);

    return current_row;
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
        return compareValues<std::string>(
            left_batch[condition.leftColumnIdx]->getString(left_row_idx),
            right_batch[condition.rightColumnIdx]->getString(right_row_idx),
            condition.op);
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
    while (left_table->hasMoreData())
    {
        left_table->readNextBatch();
        std::vector<std::shared_ptr<ColumnBatch>> left_batches = left_table->getCurrentBatch();
        while (right_table->hasMoreData())
        {
            right_table->readNextBatch();
            std::vector<std::shared_ptr<ColumnBatch>> right_batches = right_table->getCurrentBatch();
            std::vector<bool> matches(left_batches[0]->getNumRows() * right_batches[0]->getNumRows(), false);
            std::cout << "left table size: " << left_table->getCurrentBatchSize() << std::endl;
            std::cout << "right table size: " << right_table->getCurrentBatchSize() << std::endl;

            for (size_t i = 0; i < left_table->getCurrentBatchSize(); i++)
            {
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
            void **result_table_batches = allocateResultTableBatches(matches, left_batches, right_batches);
            size_t num_rows = addBatchColumns(result_table_batches, matches, left_batches, right_batches);

            std::cout << "num rows matched: " << num_rows << std::endl;
            // now the void** should be the same as the GPU result that we get
            // we need to add the result to the result table
            result_table->addResultBatch(result_table_batches, num_rows);
        }
        right_table->resetFilePositionToStart();
    }
    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    std::cout << "time taken by join duration: " << duration.count() << " seconds" << std::endl;
}

#include "./Join/read_csv.h"
#include "./Join/timer.h"
#include "device_launch_parameters.h"

void joinTablesGPU(std::shared_ptr<Table> left_table, std::shared_ptr<Table> right_table,
                   std::vector<JoinCondition> join_conditions,
                   std::shared_ptr<Table> result_table)
{
    // std::cout << "Omar join" << std::endl;
    // GpuTimer gpu_timer;
    // CSVData contents1 = readCSVWithHeader("sales1.csv");
    // CSVData contents2 = readCSVWithHeader("sales2.csv");
    // if (contents1.rows.empty())
    // {
    //     std::cerr << "Error: CSV file '" << "sales1.csv" << "' is empty or could not be read." << std::endl;
    // }
    // if (contents1.header.empty())
    // {
    //     std::cerr << "Error: CSV file has data rows but no header. Cannot identify columns." << std::endl;
    // }
    // if (contents2.rows.empty())
    // {
    //     std::cerr << "Error: CSV file '" << "sales2.csv" << "' is empty or could not be read." << std::endl;
    // }
    // if (contents2.header.empty())
    // {
    //     std::cerr << "Error: CSV file has data rows but no header. Cannot identify columns." << std::endl;
    // }
    // std::cout << "CSV Read Successfully:" << std::endl;
    // std::cout << "Header 1 columns (" << contents1.header.size() << "): ";
    // for (size_t i = 0; i < contents1.header.size(); ++i)
    // {
    //     std::cout << contents1.header[i] << (i == contents1.header.size() - 1 ? "" : ", ");
    // }
    // std::cout << std::endl;
    // std::cout << "Header 2 columns (" << contents2.header.size() << "): ";
    // for (size_t i = 0; i < contents2.header.size(); ++i)
    // {
    //     std::cout << contents2.header[i] << (i == contents2.header.size() - 1 ? "" : ", ");
    // }
    // std::cout << std::endl;

    // size_t EFFECTIVE_NUM_ROWS = contents1.rows.size();
    // size_t EFFECTIVE_NUM_ROWS_INNER = 1000;
    // size_t EFFECTIVE_NUM_COLUMNS = contents1.header.size();
    // int n_cols_out = 8;
    // int n_rows_out = 5 * (EFFECTIVE_NUM_ROWS + EFFECTIVE_NUM_ROWS_INNER);
    // DeviceStruct *h_input1 = new DeviceStruct[EFFECTIVE_NUM_COLUMNS];
    // DeviceStruct *h_input2 = new DeviceStruct[EFFECTIVE_NUM_COLUMNS];
    // DeviceStruct *h_output = new DeviceStruct[n_cols_out];
    // JoinCondition *h_join_condition = new JoinCondition[2];
    // DeviceStruct *d_input1 = nullptr;
    // DeviceStruct *d_input2 = nullptr;
    // DeviceStruct *d_output = nullptr;
    // JoinCondition *d_join_condition = nullptr;

    // h_join_condition[0].columnType = ColumnType::DATE;
    // h_join_condition[0].leftColumnIdx = 1;
    // h_join_condition[0].rightColumnIdx = 1;
    // h_join_condition[0].op = ComparisonOperator::EQUALS;
    // h_join_condition[1].columnType = ColumnType::DATE;
    // h_join_condition[1].leftColumnIdx = 2;
    // h_join_condition[1].rightColumnIdx = 2;
    // h_join_condition[1].op = ComparisonOperator::EQUALS;

    // gpu_timer.Start();
    // cudaMalloc(&d_input1, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS);
    // cudaMalloc(&d_input2, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS);
    // cudaMalloc(&d_output, sizeof(DeviceStruct) * n_cols_out);
    // cudaMalloc(&d_join_condition, sizeof(JoinCondition) * 2);
    // cudaMemcpy(d_join_condition, h_join_condition, sizeof(JoinCondition) * 2, cudaMemcpyHostToDevice);

    // for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    // {
    //     if (i != 2)
    //     {
    //         h_input1[i].type = ColumnType::DATE;
    //         h_input1[i].device_ptr = nullptr;
    //         h_input1[i].numRows = EFFECTIVE_NUM_ROWS;
    //         h_input1[i].rowSize = sizeof(int64_t) * EFFECTIVE_NUM_COLUMNS;
    //         h_input2[i].type = ColumnType::DATE;
    //         h_input2[i].device_ptr = nullptr;
    //         h_input2[i].numRows = EFFECTIVE_NUM_ROWS_INNER;
    //         h_input2[i].rowSize = sizeof(int64_t) * EFFECTIVE_NUM_COLUMNS;
    //     }
    //     else
    //     {
    //         h_input1[i].type = ColumnType::FLOAT;
    //         h_input1[i].device_ptr = nullptr;
    //         h_input1[i].numRows = EFFECTIVE_NUM_ROWS;
    //         h_input1[i].rowSize = sizeof(float) * EFFECTIVE_NUM_COLUMNS;
    //         h_input2[i].type = ColumnType::FLOAT;
    //         h_input2[i].device_ptr = nullptr;
    //         h_input2[i].numRows = EFFECTIVE_NUM_ROWS_INNER;
    //         h_input2[i].rowSize = sizeof(float) * EFFECTIVE_NUM_COLUMNS;
    //     }
    // }
    // for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    // {
    //     if (i != 2)
    //     {
    //         cudaMalloc(&h_input1[i].device_ptr, sizeof(int64_t) * EFFECTIVE_NUM_ROWS);
    //         cudaMalloc(&h_input2[i].device_ptr, sizeof(int64_t) * EFFECTIVE_NUM_ROWS_INNER);
    //         int64_t *temp_data1 = new int64_t[EFFECTIVE_NUM_ROWS];
    //         int64_t *temp_data2 = new int64_t[EFFECTIVE_NUM_ROWS_INNER];
    //         for (int j = 0; j < EFFECTIVE_NUM_ROWS; j++)
    //         {
    //             temp_data1[j] = std::stoll(contents1.rows[j][i]);
    //             if (j < EFFECTIVE_NUM_ROWS_INNER)
    //             {
    //                 temp_data2[j] = std::stoll(contents2.rows[j][i]);
    //             }
    //         }
    //         cudaMemcpy(h_input1[i].device_ptr, temp_data1, sizeof(int64_t) * EFFECTIVE_NUM_ROWS, cudaMemcpyHostToDevice);
    //         cudaMemcpy(h_input2[i].device_ptr, temp_data2, sizeof(int64_t) * EFFECTIVE_NUM_ROWS_INNER, cudaMemcpyHostToDevice);
    //         delete[] temp_data1;
    //         delete[] temp_data2;
    //     }
    //     else
    //     {
    //         cudaMalloc(&h_input1[i].device_ptr, sizeof(float) * EFFECTIVE_NUM_ROWS);
    //         cudaMalloc(&h_input2[i].device_ptr, sizeof(float) * EFFECTIVE_NUM_ROWS_INNER);
    //         float *temp_data1 = new float[EFFECTIVE_NUM_ROWS];
    //         float *temp_data2 = new float[EFFECTIVE_NUM_ROWS_INNER];

    //         for (int j = 0; j < EFFECTIVE_NUM_ROWS; j++)
    //         {
    //             temp_data1[j] = std::stof(contents1.rows[j][i]);
    //             if (j < EFFECTIVE_NUM_ROWS_INNER)
    //             {
    //                 temp_data2[j] = std::stof(contents2.rows[j][i]);
    //             }
    //         }
    //         cudaMemcpy(h_input1[i].device_ptr, temp_data1, sizeof(float) * EFFECTIVE_NUM_ROWS, cudaMemcpyHostToDevice);
    //         cudaMemcpy(h_input2[i].device_ptr, temp_data2, sizeof(float) * EFFECTIVE_NUM_ROWS_INNER, cudaMemcpyHostToDevice);
    //         delete[] temp_data1;
    //         delete[] temp_data2;
    //     }
    // }
    // cudaMemcpy(d_input1, h_input1, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_input2, h_input2, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS, cudaMemcpyHostToDevice);

    // for (int i = 0; i < n_cols_out; i++)
    // {
    //     if (i != 2)
    //     {
    //         h_output[i].type = ColumnType::DATE;
    //         h_output[i].numRows = n_rows_out;
    //         h_output[i].rowSize = sizeof(int64_t) * n_cols_out;
    //         cudaMalloc(&h_output[i].device_ptr, sizeof(int64_t) * n_rows_out);
    //     }
    //     else
    //     {
    //         h_output[i].type = ColumnType::FLOAT;
    //         h_output[i].numRows = n_rows_out;
    //         h_output[i].rowSize = sizeof(float) * n_cols_out;
    //         cudaMalloc(&h_output[i].device_ptr, sizeof(float) * n_rows_out);
    //     }
    // }
    // cudaMemcpy(d_output, h_output, sizeof(DeviceStruct) * n_cols_out, cudaMemcpyHostToDevice);

    // int shared_memory_size = sizeof(int64_t) * (EFFECTIVE_NUM_COLUMNS - 1) * EFFECTIVE_NUM_ROWS_INNER + sizeof(float) * EFFECTIVE_NUM_ROWS_INNER;
    // int offsets[4];
    // offsets[0] = 0;
    // offsets[1] = EFFECTIVE_NUM_ROWS_INNER * sizeof(int64_t);
    // offsets[2] = 2 * EFFECTIVE_NUM_ROWS_INNER * sizeof(int64_t);
    // offsets[3] = offsets[2] + sizeof(float) * EFFECTIVE_NUM_ROWS_INNER;
    // int *d_offsets = nullptr;
    // cudaMalloc(&d_offsets, sizeof(int) * 4);
    // cudaMemcpy(d_offsets, offsets, sizeof(int) * 4, cudaMemcpyHostToDevice);
    // int actual_rows_out = nested_loop_join(d_input1, d_input2, d_join_condition, EFFECTIVE_NUM_ROWS, EFFECTIVE_NUM_ROWS_INNER, 4, 4, 2, d_output, shared_memory_size, d_offsets, n_cols_out);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    // }
    // std::cout << "number of matching rows: " << actual_rows_out << std::endl;
    // DeviceStruct *h_out_temp = new DeviceStruct[n_cols_out];
    // cudaMemcpy(h_out_temp, d_output, sizeof(DeviceStruct) * n_cols_out, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n_cols_out; i++)
    // {
    //     if (i != 2)
    //     {
    //         int64_t *temp_data = new int64_t[actual_rows_out];

    //         cudaMemcpy(
    //             static_cast<int64_t *>(temp_data),
    //             static_cast<int64_t *>(h_out_temp[i].device_ptr),
    //             sizeof(int64_t) * actual_rows_out,
    //             cudaMemcpyDeviceToHost);
    //         h_output[i].device_ptr = temp_data;
    //     }
    //     else
    //     {
    //         float *temp_data = new float[actual_rows_out];
    //         cudaMemcpy(
    //             static_cast<float *>(temp_data),
    //             static_cast<float *>(h_out_temp[i].device_ptr),
    //             sizeof(float) * actual_rows_out,
    //             cudaMemcpyDeviceToHost);
    //         h_output[i].device_ptr = temp_data;
    //     }
    // }
    // gpu_timer.Stop();
    // std::cout << actual_rows_out << std::endl;
    // for (int i = 0; i < actual_rows_out; i++)
    // {
    //     if (i % 100 == 0)
    //     {
    //         std::cout << "Output Row " << i << ": ";
    //         for (int j = 0; j < n_cols_out; j++)
    //         {
    //             if (j != 2)
    //             {
    //                 std::cout << static_cast<int64_t *>(h_output[j].device_ptr)[i] << " ";
    //             }
    //             else
    //             {
    //                 std::cout << static_cast<float *>(h_output[j].device_ptr)[i] << " ";
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // std::cout << "Time taken: " << (gpu_timer.Elapsed() / 1000.0) << " s" << std::endl;
    // for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    // {
    //     // delete[] static_cast<int64_t *>(h_input1[i].device_ptr);
    //     cudaFree(h_input1[i].device_ptr);
    //     cudaFree(h_input2[i].device_ptr);
    // }
    // for (int i = 0; i < n_cols_out; i++)
    // {
    //     cudaFree(h_out_temp[i].device_ptr);
    // }
    // delete[] h_input1;
    // delete[] h_input2;
    // delete[] h_output; // But see point below about its contents
    // delete[] h_join_condition;
    // delete[] h_out_temp;
    // cudaFree(d_input1);
    // cudaFree(d_input2);
    // cudaFree(d_output);
    // cudaFree(d_join_condition);
    // cudaFree(d_offsets);
    int timeSum = 0;

    try
    {
        unsigned int global_row_count = 0;
        cudaMemcpyFromSymbol(&global_row_count, "d_global_row_count", sizeof(unsigned int));
        std::cout << "global row count: " << global_row_count << std::endl;

        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        // TODO if right is beigger then swap

        while (left_table->hasMoreData())
        {
            left_table->readNextBatch();
            std::cout << "left table size: " << left_table->getCurrentBatchSize() << std::endl;
            std::cout << "right table size: " << right_table->getCurrentBatchSize() << std::endl;
            std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
            std::vector<DeviceStruct> h_input1 = left_table->transferBatchToGPU();

            std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
            timeSum += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            while (right_table->hasMoreData())
            {
                right_table->readNextBatch();
                std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
                std::vector<DeviceStruct> h_input2 = right_table->transferBatchToGPU();
                std::vector<DeviceStruct> h_output(h_input1.size() + h_input2.size());
                int num_rows_output = (h_input1[0].numRows + h_input2[0].numRows) * 5;

                for (size_t i = 0; i < h_input1.size(); i++)
                    h_output[i] = *DeviceStruct::createStructWithoutCopy(h_input1[i].type, num_rows_output);
                for (size_t i = 0; i < h_input2.size(); i++)
                    h_output[h_input1.size() + i] = *DeviceStruct::createStructWithoutCopy(h_input2[i].type, num_rows_output);

                int *h_offsets = (int *)malloc(sizeof(int) * h_input2.size());
                h_offsets[0] = 0;
                for (size_t i = 1; i < h_input2.size(); i++)
                    h_offsets[i] = h_offsets[i - 1] + h_input2[i - 1].numRows * h_input2[i - 1].rowSize;

                // GPU memory allocation
                DeviceStruct *d_input1;
                DeviceStruct *d_input2;
                DeviceStruct *d_output;
                JoinCondition *d_join_condition;
                int *d_offsets;

                cudaMalloc(&d_input1, sizeof(DeviceStruct) * h_input1.size());
                cudaMalloc(&d_input2, sizeof(DeviceStruct) * h_input2.size());
                cudaMalloc(&d_output, sizeof(DeviceStruct) * h_output.size());
                cudaMalloc(&d_join_condition, sizeof(JoinCondition) * join_conditions.size());
                cudaMalloc(&d_offsets, sizeof(int) * h_input2.size());

                cudaMemcpy(d_input1, h_input1.data(), sizeof(DeviceStruct) * h_input1.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(d_input2, h_input2.data(), sizeof(DeviceStruct) * h_input2.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(d_output, h_output.data(), sizeof(DeviceStruct) * h_output.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(d_join_condition, join_conditions.data(), sizeof(JoinCondition) * join_conditions.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(d_offsets, h_offsets, sizeof(int) * h_input2.size(), cudaMemcpyHostToDevice);

                // TODO: calculate the shared memory size
                size_t shared_memory_size = 1000;
                size_t n_cols_out = h_input1.size() + h_input2.size();

                int actual_rows_out = nested_loop_join(d_input1, d_input2, d_join_condition, h_input1[0].numRows, h_input2[0].numRows,
                                                       h_input1.size(), h_input2.size(), join_conditions.size(),
                                                       d_output, shared_memory_size, d_offsets, n_cols_out);

                std::cout << 222222 << std::endl;
                std::cout << "number of matching rows: " << actual_rows_out << std::endl;
                DeviceStruct *h_out_temp = new DeviceStruct[n_cols_out];
                cudaMemcpy(h_out_temp, d_output, sizeof(DeviceStruct) * n_cols_out, cudaMemcpyDeviceToHost);
                void **result_table_batches = new void *[n_cols_out];
                std::cout << 333333 << std::endl;
                for (int col_idx = 0; col_idx < n_cols_out; col_idx++)
                {
                    std::cout << 333333 << std::endl;
                    std::cout << columnTypeToString(h_out_temp[col_idx].type) << std::endl;
                    switch (h_out_temp[col_idx].type)
                    {
                    case ColumnType::DATE:
                    {
                        std::cout << 444444 << std::endl;
                        int64_t *temp_data = new int64_t[actual_rows_out];
                        cudaMemcpy(
                            temp_data,
                            h_out_temp[col_idx].device_ptr,
                            sizeof(int64_t) * actual_rows_out,
                            cudaMemcpyDeviceToHost);
                        result_table_batches[col_idx] = temp_data;
                        std::cout << 555555 << std::endl;
                        break;
                    }
                    case ColumnType::FLOAT:
                    {
                        std::cout << 666666 << std::endl;
                        float *temp_data = new float[actual_rows_out];
                        cudaMemcpy(
                            temp_data,
                            h_out_temp[col_idx].device_ptr,
                            sizeof(float) * actual_rows_out,
                            cudaMemcpyDeviceToHost);
                        result_table_batches[col_idx] = temp_data;
                        std::cout << 777777 << std::endl;
                        break;
                    }
                    default:
                        throw "Invalid column type";
                    }
                }
                // TODO: copy data correctly
                std::cout << "num rows matched: " << actual_rows_out << std::endl;
                result_table->addResultBatch(result_table_batches, actual_rows_out);
                delete[] h_out_temp;
                delete[] result_table_batches;

                cudaFree(d_input1);
                cudaFree(d_input2);
                cudaFree(d_output);
                cudaFree(d_join_condition);
                cudaFree(d_offsets);

                for (size_t i = 0; i < h_input2.size(); i++)
                    DeviceStruct::deleteStruct(h_input2[i]); // deletes innner array pointer
                for (size_t i = 0; i < h_output.size(); i++)
                    DeviceStruct::deleteStruct(h_output[i]);

                unsigned int zero = 0;
                std::cout << 333333 << std::endl;
                cudaMemcpyToSymbol("d_global_row_count", &zero, sizeof(unsigned int));
                std::cout << 444444 << std::endl;
                std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
                timeSum += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();    
            }

             end_time = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < h_input1.size(); i++)
                DeviceStruct::deleteStruct(h_input1[i]); // deletes innner array pointer
            end_time = std::chrono::high_resolution_clock::now();
            timeSum += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            right_table->resetFilePositionToStart();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
   
    std::cout << "time taken by join duration: " << timeSum << " milliseconds" << std::endl;
}