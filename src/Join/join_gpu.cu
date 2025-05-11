#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../headers/table.h"
#include "../headers/device_struct.h"
#include "../headers/enums.h"
#include "join.h"
#include "operators/join.h"
#include "globals.cuh"
#include <chrono>
#include <iostream>
#include "../headers/constants.h"
void joinTablesGPU(std::shared_ptr<Table> left_table, std::shared_ptr<Table> right_table,
                   std::vector<JoinCondition> join_conditions,
                   std::shared_ptr<Table> result_table)
{
    int timeSum = 0;

    try
    {
        unsigned int global_row_count = 0;
        cudaMemcpyFromSymbol(&global_row_count, d_global_row_count, sizeof(unsigned int));
        std::cout << "global row count: " << global_row_count << std::endl;

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
                right_table->readNextBatch(500);
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
                {
                    int offset = ((h_input2[i - 1].numRows * h_input2[i - 1].rowSize) + 7) & ~7;
                    h_offsets[i] = h_offsets[i - 1] + offset;
                }
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
                size_t shared_memory_size = h_offsets[h_input2.size() - 1] +
                                            h_input2[h_input2.size() - 1].rowSize * h_input2[h_input2.size() - 1].numRows;
                // std::cout << "shared memory size: " << shared_memory_size << std::endl;
                size_t n_cols_out = h_input1.size() + h_input2.size();
                // printf("nrows 1 %d\n", (int)h_input1[0].numRows);
                // printf("nrows 2 %d\n", (int)h_input2[0].numRows);
                // std::cout << "Table 2 (" << right_table->getName() << ") has " << (int)h_input2[0].numRows << " rows" << std::endl;
                // printf("hello world");
                int actual_rows_out = nested_loop_join(d_input1, d_input2, d_join_condition, h_input1[0].numRows, h_input2[0].numRows,
                                                       h_input1.size(), h_input2.size(), join_conditions.size(),
                                                       d_output, shared_memory_size, d_offsets, n_cols_out);

                // std::cout << "number of matching rows: " << actual_rows_out << std::endl;
                DeviceStruct *h_out_temp = new DeviceStruct[n_cols_out];
                cudaMemcpy(h_out_temp, d_output, sizeof(DeviceStruct) * n_cols_out, cudaMemcpyDeviceToHost);
                void **result_table_batches = new void *[n_cols_out];

                for (int col_idx = 0; col_idx < n_cols_out; col_idx++)
                {
                    // std::cout << "column type: " << columnTypeToString(h_out_temp[col_idx].type) << std::endl;
                    switch (h_out_temp[col_idx].type)
                    {
                    case ColumnType::DATE:
                    {
                        int64_t *temp_data = new int64_t[actual_rows_out];
                        cudaMemcpy(
                            temp_data,
                            h_out_temp[col_idx].device_ptr,
                            sizeof(int64_t) * actual_rows_out,
                            cudaMemcpyDeviceToHost);
                        result_table_batches[col_idx] = temp_data;
                        break;
                    }
                    case ColumnType::FLOAT:
                    {
                        float *temp_data = new float[actual_rows_out];
                        cudaMemcpy(
                            temp_data,
                            h_out_temp[col_idx].device_ptr,
                            sizeof(float) * actual_rows_out,
                            cudaMemcpyDeviceToHost);
                        result_table_batches[col_idx] = temp_data;
                        break;
                    }
                    case ColumnType::STRING:
                    {
                        char *temp_data = new char[actual_rows_out * MAX_STRING_LENGTH];
                        cudaMemcpy(
                            temp_data,
                            h_out_temp[col_idx].device_ptr,
                            sizeof(char) * actual_rows_out * MAX_STRING_LENGTH,
                            cudaMemcpyDeviceToHost);
                        result_table_batches[col_idx] = temp_data;
                        break;
                    }
                    default:
                        throw "Invalid column type";
                    }
                }

                // std::cout << "num rows matched: " << actual_rows_out << std::endl;
                result_table->addResultBatch(result_table_batches, actual_rows_out);
                delete[] h_out_temp;

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
                cudaMemcpyToSymbol(d_global_row_count, &zero, sizeof(unsigned int));

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