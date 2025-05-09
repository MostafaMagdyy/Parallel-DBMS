#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include "sort.h"
#include "utils.h"
#include "device_struct.h"
#include "timer.h"
#include < chrono >
#include "read_csv.h" // Must be first for precompiled headers, or just good practice
#include <fstream>    // For std::ifstream
#include <sstream>    // For std::stringstream
#define NUM_ELEMENTS (1 << 20)

int main()
{

    // Use num_elements_from_csv as the effective NUM_ELEMENTS for this run
    GpuTimer gpu_timer;
    srand(1);
    std::cout << "Simulating array of DeviceStructs...\n";

    // Simulate a single column (FLOAT) as if received
    float *h_col = new float[NUM_ELEMENTS];
    float *h_out = new float[NUM_ELEMENTS];

    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        h_col[i] = static_cast<float>(std::rand() % 1000000) / 123.431312f;
    }

    // Allocate device memory for the FLOAT column
    float *d_col = nullptr;
    DeviceStruct *d_input_structs = nullptr;
    const int input_col_count = 1;
    gpu_timer.Start();
    checkCudaErrors(cudaMalloc(&d_col, sizeof(float) * NUM_ELEMENTS));
    checkCudaErrors(cudaMalloc(&d_input_structs, sizeof(DeviceStruct) * input_col_count));
    // Create input array of DeviceStructs (simulate 1 column received)
    DeviceStruct *input_structs = new DeviceStruct[input_col_count];
    input_structs[0].type = ColumnType::FLOAT;
    input_structs[0].device_ptr = d_col;
    input_structs[0].numRows = NUM_ELEMENTS;
    input_structs[0].rowSize = sizeof(float);

    checkCudaErrors(cudaMemcpy(d_col, h_col, sizeof(float) * NUM_ELEMENTS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_input_structs, input_structs, sizeof(DeviceStruct) * input_col_count, cudaMemcpyHostToDevice));

    // Create extra transformed column (UINT64)
    uint64_t *d_transformed_col = nullptr;
    DeviceStruct *d_transformed_col_struct = nullptr;

    checkCudaErrors(cudaMalloc(&d_transformed_col, sizeof(uint64_t) * NUM_ELEMENTS));
    checkCudaErrors(cudaMalloc(&d_transformed_col_struct, sizeof(DeviceStruct)));

    DeviceStruct transformed_struct;
    transformed_struct.type = ColumnType::UINT64;
    transformed_struct.device_ptr = d_transformed_col;
    transformed_struct.numRows = NUM_ELEMENTS;
    transformed_struct.rowSize = sizeof(uint64_t);

    checkCudaErrors(cudaMemcpy(d_transformed_col_struct, &transformed_struct, sizeof(DeviceStruct), cudaMemcpyHostToDevice));

    float *d_col_output = nullptr;
    DeviceStruct *d_output_structs = nullptr;

    checkCudaErrors(cudaMalloc(&d_col_output, sizeof(float) * NUM_ELEMENTS));
    checkCudaErrors(cudaMalloc(&d_output_structs, sizeof(DeviceStruct) * (input_col_count + 1)));
    // Create output array of DeviceStructs (same size as input, does not include transformed)
    DeviceStruct *output_structs = new DeviceStruct[input_col_count + 1];
    // for (int i = 0; i < input_col_count + 1; i++)
    // {
    // output_structs[i].type = input_structs[i].type; // Will be set later
    // output_structs[i].device_ptr = nullptr; // Will be allocated later if needed
    // output_structs[i].numRows = NUM_ELEMENTS;
    // output_structs[i].rowSize = 0;
    // }
    output_structs[0].type = ColumnType::FLOAT;
    output_structs[0].device_ptr = d_col_output;
    output_structs[0].numRows = NUM_ELEMENTS;
    output_structs[0].rowSize = sizeof(float);

    output_structs[1].type = ColumnType::UINT64;
    output_structs[1].device_ptr = d_transformed_col;
    output_structs[1].numRows = NUM_ELEMENTS;
    output_structs[1].rowSize = sizeof(uint64_t);
    checkCudaErrors(cudaMemcpy(d_output_structs, output_structs, sizeof(DeviceStruct) * (input_col_count + 1), cudaMemcpyHostToDevice));
    // radix_sort(d_output_structs, d_input_structs, NUM_ELEMENTS, 0, input_col_count + 1);
    checkCudaErrors(cudaMemcpy(h_out, d_col, sizeof(float) * NUM_ELEMENTS, cudaMemcpyDeviceToHost));
    gpu_timer.Stop();
    double gpu_time = (gpu_timer.Elapsed() / 1000.0);
    std::cout << "GPU execution time: " << gpu_time << " seconds\n";
    std::vector<float> h_in_sorted(h_col, h_col + NUM_ELEMENTS);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(h_in_sorted.begin(), h_in_sorted.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU execution time: " << cpu_duration.count() << " seconds\n";
    double speedup = cpu_duration.count() / gpu_time;
    std::cout << "Speedup: " << speedup << "x\n";

    bool is_correct = true;
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        // if (i < 1000000 && i > 999000)
        // {
        // std::cout << "GPU: " << h_out[i] << ", CPU: " << h_in_sorted[i] << std::endl;
        // }
        if (h_out[i] != h_in_sorted[i])
        {
            std::cerr << "Mismatch at index " << i
                      << ": GPU = " << h_out[i]
                      << ", CPU = " << h_in_sorted[i] << std::endl;
            is_correct = false;
            break;
        }
    }

    if (is_correct)
    {
        std::cout << "GPU radix sort matches std::sort." << std::endl;
    }
    else
    {
        std::cerr << "GPU radix sort does NOT match std::sort." << std::endl;
    }

    // Cleanup host memory
    delete[] h_col;
    delete[] h_out;

    // Free only if you're not doing further processing
    checkCudaErrors(cudaFree(d_col));
    checkCudaErrors(cudaFree(d_transformed_col));

    checkCudaErrors(cudaFree(d_input_structs));
    checkCudaErrors(cudaFree(d_transformed_col_struct));
    checkCudaErrors(cudaFree(d_col_output));
    checkCudaErrors(cudaFree(d_output_structs));

    delete[] input_structs;
    delete[] output_structs;

    std::cout << "DeviceStruct simulation complete.\n";

    return 0;
}
