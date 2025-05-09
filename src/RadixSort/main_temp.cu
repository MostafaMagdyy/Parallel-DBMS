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

#include "../operators/sort.h"
#include "utils.h"
#include "../headers/device_struct.h"
#include "timer.h"
#include <chrono>
#include "read_csv.h"

int main()
{
    GpuTimer gpu_timer;

    CSVData contents = readCSVWithHeader("sales.csv");

    if (contents.rows.empty())
    {
        std::cerr << "Error: CSV file '" << "sales.csv" << "' is empty or could not be read." << std::endl;
        return 1;
    }
    if (contents.header.empty())
    {
        std::cerr << "Error: CSV file has data rows but no header. Cannot identify columns." << std::endl;
        return 1;
    }

    std::cout << "CSV Read Successfully:" << std::endl;
    std::cout << "Header columns (" << contents.header.size() << "): ";
    for (size_t i = 0; i < contents.header.size(); ++i)
    {
        std::cout << contents.header[i] << (i == contents.header.size() - 1 ? "" : ", ");
    }
    std::cout << std::endl;

    size_t EFFECTIVE_NUM_ROWS = 1 << 20;
    size_t EFFECTIVE_NUM_COLUMNS = contents.header.size();

    std::cout << "Number of data rows: " << contents.rows.size() << " Cols: " << EFFECTIVE_NUM_COLUMNS << std::endl;

    DeviceStruct *h_input = new DeviceStruct[EFFECTIVE_NUM_COLUMNS];
    DeviceStruct *h_output = new DeviceStruct[EFFECTIVE_NUM_COLUMNS + 1];
    DeviceStruct *d_input = nullptr;
    DeviceStruct *d_output = nullptr;
    gpu_timer.Start();
    checkCudaErrors(cudaMalloc(&d_input, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS));
    checkCudaErrors(cudaMalloc(&d_output, sizeof(DeviceStruct) * (EFFECTIVE_NUM_COLUMNS + 1)));

    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        h_input[i].type = ColumnType::DATE;
        h_input[i].device_ptr = nullptr;
        h_input[i].numRows = EFFECTIVE_NUM_ROWS;
        h_input[i].rowSize = sizeof(int64_t) * EFFECTIVE_NUM_COLUMNS;
    }

    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        h_input[i].device_ptr = new int64_t[EFFECTIVE_NUM_ROWS];
        checkCudaErrors(cudaMalloc(&h_input[i].device_ptr, sizeof(int64_t) * EFFECTIVE_NUM_ROWS));
        int64_t *temp_data = new int64_t[EFFECTIVE_NUM_ROWS];
        for (int j = 0; j < EFFECTIVE_NUM_ROWS; j++)
        {
            temp_data[j] = std::stoll(contents.rows[j][i]);
        }
        checkCudaErrors(cudaMemcpy(h_input[i].device_ptr, temp_data, sizeof(int64_t) * EFFECTIVE_NUM_ROWS, cudaMemcpyHostToDevice));
        delete[] temp_data;
    }
    checkCudaErrors(cudaMemcpy(d_input, h_input, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS, cudaMemcpyHostToDevice));

    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        h_output[i].type = ColumnType::DATE;
        h_output[i].device_ptr = nullptr;
        h_output[i].numRows = EFFECTIVE_NUM_ROWS;
        h_output[i].rowSize = sizeof(int64_t) * EFFECTIVE_NUM_COLUMNS;
    }
    h_output[EFFECTIVE_NUM_COLUMNS].type = ColumnType::DATE;
    h_output[EFFECTIVE_NUM_COLUMNS].device_ptr = nullptr;
    h_output[EFFECTIVE_NUM_COLUMNS].numRows = EFFECTIVE_NUM_ROWS;
    h_output[EFFECTIVE_NUM_COLUMNS].rowSize = sizeof(uint64_t) * EFFECTIVE_NUM_COLUMNS;

    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        int64_t *temp_data = new int64_t[EFFECTIVE_NUM_ROWS];
        checkCudaErrors(cudaMalloc(&h_output[i].device_ptr, sizeof(int64_t) * EFFECTIVE_NUM_ROWS));
        checkCudaErrors(cudaMemcpy(h_output[i].device_ptr, temp_data, sizeof(int64_t) * EFFECTIVE_NUM_ROWS, cudaMemcpyHostToDevice));
        delete[] temp_data;
    }
    uint64_t *temp_data = new uint64_t[EFFECTIVE_NUM_ROWS];
    checkCudaErrors(cudaMalloc(&h_output[EFFECTIVE_NUM_COLUMNS].device_ptr, sizeof(uint64_t) * EFFECTIVE_NUM_ROWS));
    checkCudaErrors(cudaMemcpy(h_output[EFFECTIVE_NUM_COLUMNS].device_ptr, temp_data, sizeof(uint64_t) * EFFECTIVE_NUM_ROWS, cudaMemcpyHostToDevice));
    delete[] temp_data;
    checkCudaErrors(cudaMemcpy(d_output, h_output, sizeof(DeviceStruct) * (EFFECTIVE_NUM_COLUMNS + 1), cudaMemcpyHostToDevice));

    radix_sort(d_output, d_input, EFFECTIVE_NUM_ROWS, 1, EFFECTIVE_NUM_COLUMNS + 1);
    DeviceStruct *h_final_out = new DeviceStruct[EFFECTIVE_NUM_COLUMNS];
    checkCudaErrors(cudaMemcpy(h_final_out, d_input, sizeof(DeviceStruct) * EFFECTIVE_NUM_COLUMNS, cudaMemcpyDeviceToHost));
    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        int64_t *temp_data = new int64_t[EFFECTIVE_NUM_ROWS];

        checkCudaErrors(cudaMemcpy(
            static_cast<int64_t *>(temp_data),
            static_cast<int64_t *>(h_final_out[i].device_ptr),
            sizeof(int64_t) * EFFECTIVE_NUM_ROWS,
            cudaMemcpyDeviceToHost));
        h_final_out[i].device_ptr = temp_data;
    }
    gpu_timer.Stop();
    double GPU_time = gpu_timer.Elapsed() / 1000.0;
    std::vector<std::vector<int64_t>> h_cpu_sorted_expected(EFFECTIVE_NUM_ROWS, std::vector<int64_t>(EFFECTIVE_NUM_COLUMNS));
    // std::cout << "hello world" << std::endl;
    for (int i = 0; i < EFFECTIVE_NUM_ROWS; i++)
    {
        for (int j = 0; j < EFFECTIVE_NUM_COLUMNS; j++)
        {
            h_cpu_sorted_expected[i][j] = std::stoll(contents.rows[i][j]);
        }
    }
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::stable_sort(
        h_cpu_sorted_expected.begin(),
        h_cpu_sorted_expected.end(),
        [](const std::vector<int64_t> &a, const std::vector<int64_t> &b)
        {
            return a[1] < b[1]; // sort based on column index 1
        });
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    bool correct = true;
    // for (int i = 0; i < 100; i++)
    // {
    //     for (int j = 0; j < EFFECTIVE_NUM_COLUMNS; j++)
    //     {
    //         // std::cout << static_cast<int64_t *>(h_final_out[j].device_ptr)[i] << ",";
    //         std::cout << h_cpu_sorted_expected[i][j] << ",";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "-------------------------------------------------------------------" << std::endl;
    // }
    // for (int i = 0; i < EFFECTIVE_NUM_ROWS; i++)
    // {
    //     for (int j = 0; j < EFFECTIVE_NUM_COLUMNS; j++)
    //     {
    //         // std::cout << static_cast<int64_t *>(h_final_out[j].device_ptr)[i] << ",";
    //         std::cout << h_cpu_sorted_expected[i][j] << ",";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "-------------------------------------------------------------------" << std::endl;
    // }
    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        for (int j = 0; j < EFFECTIVE_NUM_ROWS; j++)
        {
            if (h_cpu_sorted_expected[j][i] != static_cast<int64_t *>(h_final_out[i].device_ptr)[j])
            {
                std::cerr << "Mismatch at index " << i << ", " << j
                          << ": GPU = " << static_cast<int64_t *>(h_final_out[i].device_ptr)[j]
                          << ", CPU = " << h_cpu_sorted_expected[j][i] << std::endl;
                correct = false;
            }
        }
    }
    if (correct)
    {
        std::cout << "GPU radix sort matches CPU sort." << std::endl;
    }
    else
    {
        std::cerr << "GPU radix sort does NOT match CPU sort." << std::endl;
    }
    std::cout << "GPU execution time: " << GPU_time << " seconds\n";
    std::cout << "CPU execution time: " << cpu_duration.count() << " seconds\n";
    std::cout << "Speedup: " << cpu_duration.count() / GPU_time << "x\n";
    for (int i = 0; i < EFFECTIVE_NUM_COLUMNS; i++)
    {
        cudaFree(d_input[i].device_ptr);
        cudaFree(d_output[i].device_ptr);
        free(h_final_out[i].device_ptr);
    }
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    delete[] h_final_out;

    return 0;
}