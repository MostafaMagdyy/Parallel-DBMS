#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <vector>
#include "sort.h"
#include "utils.h"
#include <chrono>
#include <random>

#include "timer.h"
// threshold is 1<<16
#define NUM_ELEMENTS (1 << 20)

std::mt19937_64 rng(1); // fixed seed for reproducibility
std::uniform_int_distribution<uint64_t> dist;

uint64_t rand64()
{
    return dist(rng);
}
int main()
{
    GpuTimer gpu_timer;
    srand(1);
    std::cout << "h_in size: " << NUM_ELEMENTS << std::endl;

    float *h_in = new float[NUM_ELEMENTS];
    float *h_out_gpu = new float[NUM_ELEMENTS];

    for (int j = 0; j < NUM_ELEMENTS; j++)
    {

        h_in[j] = static_cast<float>(std::rand() % 1000000) / 123.431312f; // e.g., values from 0.0 to 999.9
    }

    uint64_t *d_in = nullptr;
    uint64_t *d_out = nullptr;

    // PREPARE INPUT
    uint64_t *h_input_uint = new uint64_t[NUM_ELEMENTS];
    if (!h_input_uint)
    { /* handle allocation error */
        return 1;
    }
    uint64_t *h_output_uint = new uint64_t[NUM_ELEMENTS];
    if (!h_output_uint)
    { /* handle allocation error */
        return 1;
    }

    auto hostFloatToOrderedInt = [](float f) -> uint64_t
    {
        uint64_t x = *reinterpret_cast<uint64_t *>(&f);
        return (x & 0x80000000) ? ~x : x ^ 0x80000000;
    };
    auto hostOrderedIntToFloat = [](uint64_t ui) -> float
    {
        ui = (ui & 0x80000000) ? ui ^ 0x80000000 : ~ui; // Logic from device function
        return *reinterpret_cast<float *>(&ui);
    };

    gpu_timer.Start();

    // Conert the input float array to uint64_t
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        h_input_uint[i] = hostFloatToOrderedInt(h_in[i]);
    }
    // END CONVERT

    checkCudaErrors(cudaMalloc(&d_in, sizeof(uint64_t) * NUM_ELEMENTS));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(uint64_t) * NUM_ELEMENTS));

    checkCudaErrors(cudaMemcpy(d_in, h_input_uint, sizeof(uint64_t) * NUM_ELEMENTS, cudaMemcpyHostToDevice));
    radix_sort(d_out, d_in, NUM_ELEMENTS);

    checkCudaErrors(cudaMemcpy(h_output_uint, d_out, sizeof(uint64_t) * NUM_ELEMENTS, cudaMemcpyDeviceToHost));

    // Convert the output uint64_t array back to float
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        h_out_gpu[i] = hostOrderedIntToFloat(h_output_uint[i]);
    }
    // END CONVERT
    gpu_timer.Stop();
    delete[] h_input_uint;  // Free the host temporary array AFTER copying
    delete[] h_output_uint; // Free the temporary host uint array now that conversion is done
    double gpu_time = (gpu_timer.Elapsed() / 1000.0);
    std::cout << "GPU execution time: " << gpu_time << " seconds\n";
    std::vector<float> h_in_sorted(h_in, h_in + NUM_ELEMENTS);
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

        if (h_out_gpu[i] != h_in_sorted[i])
        {
            std::cerr << "Mismatch at index " << i
                      << ": GPU = " << h_out_gpu[i]
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
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_in));

    delete[] h_in;
    delete[] h_out_gpu;
    std::cout << "Sort completed successfully." << std::endl;
    return 0;
}
