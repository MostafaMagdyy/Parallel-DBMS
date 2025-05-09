#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <numeric>

#include "utils.h"
#include "device_struct.h"
#include "timer.h"
#include <chrono>
#include "read_csv.h"
#include "join.h"
#include "globals.cuh"
int main()
{
    GpuTimer gpu_timer;
    CSVData contents1 = readCSVWithHeader("sales1.csv");
    CSVData contents2 = readCSVWithHeader("sales2.csv");
    if (contents1.rows.empty())
    {
        std::cerr << "Error: CSV file '" << "sales1.csv" << "' is empty or could not be read." << std::endl;
        return 1;
    }
    if (contents1.header.empty())
    {
        std::cerr << "Error: CSV file has data rows but no header. Cannot identify columns." << std::endl;
        return 1;
    }
    if (contents2.rows.empty())
    {
        std::cerr << "Error: CSV file '" << "sales2.csv" << "' is empty or could not be read." << std::endl;
        return 1;
    }
    if (contents2.header.empty())
    {
        std::cerr << "Error: CSV file has data rows but no header. Cannot identify columns." << std::endl;
        return 1;
    }
    std::cout << "CSV Read Successfully:" << std::endl;
    std::cout << "Header 1 columns (" << contents1.header.size() << "): ";
    for (size_t i = 0; i < contents1.header.size(); ++i)
    {
        std::cout << contents1.header[i] << (i == contents1.header.size() - 1 ? "" : ", ");
    }
    std::cout << std::endl;
    std::cout << "Header 2 columns (" << contents2.header.size() << "): ";
    for (size_t i = 0; i < contents2.header.size(); ++i)
    {
        std::cout << contents2.header[i] << (i == contents2.header.size() - 1 ? "" : ", ");
    }
    std::cout << std::endl;

    size_t EFFECTIVE_NUM_ROWS = contents1.rows.size();
    size_t EFFECTIVE_NUM_ROWS_INNER = 1000;
    size_t EFFECTIVE_NUM_COLUMNS = contents1.header.size();
    int n_cols_out = 6;
    int n_rows_out = 5 * (EFFECTIVE_NUM_ROWS + EFFECTIVE_NUM_ROWS_INNER);
    DeviceStruct *h_input1 = new DeviceStruct[EFFECTIVE_NUM_COLUMNS];
    DeviceStruct *h_input2 = new DeviceStruct[EFFECTIVE_NUM_COLUMNS];
    DeviceStruct *h_output = new DeviceStruct[n_cols_out];
    JoinCondition *h_join_condition = new JoinCondition[2];
    DeviceStruct *d_input1 = nullptr;
    DeviceStruct *d_input2 = nullptr;
    DeviceStruct *d_output = nullptr;
    JoinCondition *d_join_condition = nullptr;

    h_join_condition[0].columnType = ColumnType::INT;
    h_join_condition[0].leftColumnIdx = 1;
    h_join_condition[0].rightColumnIdx = 1;
    h_join_condition[0].op = ComparisonOperator::EQUALS;
    h_join_condition[1].columnType = ColumnType::FLOAT;
    h_join_condition[1].leftColumnIdx = 2;
    h_join_condition[1].rightColumnIdx = 2;
    h_join_condition[1].op = ComparisonOperator::EQUALS;

    gpu_timer.Start();
    checkCudaErrors(cudaMalloc(&d_input1, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS));
    checkCudaErrors(cudaMalloc(&d_input2, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS));
    checkCudaErrors(cudaMalloc(&d_output, sizeof(DeviceStruct) * n_cols_out));
    checkCudaErrors(cudaMalloc(&d_join_condition, sizeof(JoinCondition) * 2));
    checkCudaErrors(cudaMemcpy(d_join_condition, h_join_condition, sizeof(JoinCondition) * 2, cudaMemcpyHostToDevice));

    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        if (i != 2)
        {
            h_input1[i].type = ColumnType::INT;
            h_input1[i].device_ptr = nullptr;
            h_input1[i].numRows = EFFECTIVE_NUM_ROWS;
            h_input1[i].rowSize = sizeof(int64_t) * EFFECTIVE_NUM_COLUMNS;
            h_input2[i].type = ColumnType::INT;
            h_input2[i].device_ptr = nullptr;
            h_input2[i].numRows = EFFECTIVE_NUM_ROWS_INNER;
            h_input2[i].rowSize = sizeof(int64_t) * EFFECTIVE_NUM_COLUMNS;
        }
        else
        {
            h_input1[i].type = ColumnType::FLOAT;
            h_input1[i].device_ptr = nullptr;
            h_input1[i].numRows = EFFECTIVE_NUM_ROWS;
            h_input1[i].rowSize = sizeof(float) * EFFECTIVE_NUM_COLUMNS;
            h_input2[i].type = ColumnType::FLOAT;
            h_input2[i].device_ptr = nullptr;
            h_input2[i].numRows = EFFECTIVE_NUM_ROWS_INNER;
            h_input2[i].rowSize = sizeof(float) * EFFECTIVE_NUM_COLUMNS;
        }
    }
    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        if (i != 2)
        {
            checkCudaErrors(cudaMalloc(&h_input1[i].device_ptr, sizeof(int64_t) * EFFECTIVE_NUM_ROWS));
            checkCudaErrors(cudaMalloc(&h_input2[i].device_ptr, sizeof(int64_t) * EFFECTIVE_NUM_ROWS_INNER));
            int64_t *temp_data1 = new int64_t[EFFECTIVE_NUM_ROWS];
            int64_t *temp_data2 = new int64_t[EFFECTIVE_NUM_ROWS_INNER];
            for (int j = 0; j < EFFECTIVE_NUM_ROWS; j++)
            {
                temp_data1[j] = std::stoll(contents1.rows[j][i]);
                if (j < EFFECTIVE_NUM_ROWS_INNER)
                {
                    temp_data2[j] = std::stoll(contents2.rows[j][i]);
                }
            }
            checkCudaErrors(cudaMemcpy(h_input1[i].device_ptr, temp_data1, sizeof(int64_t) * EFFECTIVE_NUM_ROWS, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(h_input2[i].device_ptr, temp_data2, sizeof(int64_t) * EFFECTIVE_NUM_ROWS_INNER, cudaMemcpyHostToDevice));
            delete[] temp_data1;
            delete[] temp_data2;
        }
        else
        {
            checkCudaErrors(cudaMalloc(&h_input1[i].device_ptr, sizeof(float) * EFFECTIVE_NUM_ROWS));
            checkCudaErrors(cudaMalloc(&h_input2[i].device_ptr, sizeof(float) * EFFECTIVE_NUM_ROWS_INNER));
            float *temp_data1 = new float[EFFECTIVE_NUM_ROWS];
            float *temp_data2 = new float[EFFECTIVE_NUM_ROWS_INNER];

            for (int j = 0; j < EFFECTIVE_NUM_ROWS; j++)
            {
                temp_data1[j] = std::stof(contents1.rows[j][i]);
                if (j < EFFECTIVE_NUM_ROWS_INNER)
                {
                    temp_data2[j] = std::stof(contents2.rows[j][i]);
                }
            }
            checkCudaErrors(cudaMemcpy(h_input1[i].device_ptr, temp_data1, sizeof(float) * EFFECTIVE_NUM_ROWS, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(h_input2[i].device_ptr, temp_data2, sizeof(float) * EFFECTIVE_NUM_ROWS_INNER, cudaMemcpyHostToDevice));
            delete[] temp_data1;
            delete[] temp_data2;
        }
    }
    checkCudaErrors(cudaMemcpy(d_input1, h_input1, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_input2, h_input2, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS, cudaMemcpyHostToDevice));

    for (int i = 0; i < n_cols_out; i++)
    {
        if (i != 2)
        {
            h_output[i].type = ColumnType::INT;
            h_output[i].numRows = n_rows_out;
            h_output[i].rowSize = sizeof(int64_t) * n_cols_out;
            checkCudaErrors(cudaMalloc(&h_output[i].device_ptr, sizeof(int64_t) * n_rows_out));
        }
        else
        {
            h_output[i].type = ColumnType::FLOAT;
            h_output[i].numRows = n_rows_out;
            h_output[i].rowSize = sizeof(float) * n_cols_out;
            checkCudaErrors(cudaMalloc(&h_output[i].device_ptr, sizeof(float) * n_rows_out));
        }
    }
    checkCudaErrors(cudaMemcpy(d_output, h_output, sizeof(DeviceStruct) * n_cols_out, cudaMemcpyHostToDevice));

    int shared_memory_size = sizeof(int64_t) * (EFFECTIVE_NUM_COLUMNS - 1) * EFFECTIVE_NUM_ROWS_INNER + sizeof(float) * EFFECTIVE_NUM_ROWS_INNER;
    int offsets[4];
    offsets[0] = 0;
    offsets[1] = EFFECTIVE_NUM_ROWS_INNER * sizeof(int64_t);
    offsets[2] = 2 * EFFECTIVE_NUM_ROWS_INNER * sizeof(int64_t);
    offsets[3] = offsets[2] + sizeof(float) * EFFECTIVE_NUM_ROWS_INNER;
    int *d_offsets = nullptr;
    checkCudaErrors(cudaMalloc(&d_offsets, sizeof(int) * 4));
    checkCudaErrors(cudaMemcpy(d_offsets, offsets, sizeof(int) * 4, cudaMemcpyHostToDevice));
    nested_loop_join(d_input1, d_input2, d_join_condition, EFFECTIVE_NUM_ROWS, EFFECTIVE_NUM_ROWS_INNER, 4, 4, 2, d_output, shared_memory_size, d_offsets, n_cols_out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    unsigned int actual_rows_out;
    checkCudaErrors(cudaMemcpyFromSymbol(&actual_rows_out, d_global_row_count, sizeof(unsigned int)));
    DeviceStruct *h_out_temp = new DeviceStruct[n_cols_out];
    checkCudaErrors(cudaMemcpy(h_out_temp, d_output, sizeof(DeviceStruct) * n_cols_out, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_cols_out; i++)
    {
        if (i != 2)
        {
            int64_t *temp_data = new int64_t[actual_rows_out];

            checkCudaErrors(cudaMemcpy(
                static_cast<int64_t *>(temp_data),
                static_cast<int64_t *>(h_out_temp[i].device_ptr),
                sizeof(int64_t) * actual_rows_out,
                cudaMemcpyDeviceToHost));
            h_output[i].device_ptr = temp_data;
        }
        else
        {
            float *temp_data = new float[actual_rows_out];
            checkCudaErrors(cudaMemcpy(
                static_cast<float *>(temp_data),
                static_cast<float *>(h_out_temp[i].device_ptr),
                sizeof(float) * actual_rows_out,
                cudaMemcpyDeviceToHost));
            h_output[i].device_ptr = temp_data;
        }
    }
    gpu_timer.Stop();
    std::cout << actual_rows_out << std::endl;
    for (int i = 0; i < actual_rows_out; i++)
    {
        if (i % 100 == 0)
        {
            std::cout << "Output Row " << i << ": ";
            for (int j = 0; j < n_cols_out; j++)
            {
                if (j != 2)
                {
                    std::cout << static_cast<int64_t *>(h_output[j].device_ptr)[i] << " ";
                }
                else
                {
                    std::cout << static_cast<float *>(h_output[j].device_ptr)[i] << " ";
                }
            }
            std::cout << std::endl;
        }
    }

    std::cout << "Time taken: " << (gpu_timer.Elapsed() / 1000.0) << " s" << std::endl;
    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        // delete[] static_cast<int64_t *>(h_input1[i].device_ptr);
        cudaFree(h_input1[i].device_ptr);
        cudaFree(h_input2[i].device_ptr);
    }
    for (int i = 0; i < n_cols_out; i++)
    {
        cudaFree(h_out_temp[i].device_ptr);
    }
    delete[] h_input1;
    delete[] h_input2;
    delete[] h_output; // But see point below about its contents
    delete[] h_join_condition;
    delete[] h_out_temp;
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    cudaFree(d_join_condition);
    cudaFree(d_offsets);
    return 0;
}