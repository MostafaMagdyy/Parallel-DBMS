#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <vector>
#include <chrono>
#include <random>
#include <cstdint> // For int64_t, uint64_t

#include "sort.h"  // Assuming this contains the radix_sort declaration
#include "utils.h" // Assuming this contains checkCudaErrors
#include "timer.h" // Assuming this contains GpuTimer

// threshold is 1<<16 - This comment might be less relevant now or needs update depending on radix_sort implementation
#define NUM_ELEMENTS (1 << 20)

std::mt19937_64 rng(std::random_device{}()); // 64-bit Mersenne Twister
std::uniform_int_distribution<int64_t> dist(INT64_MIN, INT64_MAX);

int main()
{
    GpuTimer gpu_timer;
    srand(1); // Seed legacy rand() for consistency if used
    std::cout << "h_in size: " << NUM_ELEMENTS << std::endl;

    // --- Host array declarations ---
    int64_t *h_in = new int64_t[NUM_ELEMENTS];
    int64_t *h_out_gpu = new int64_t[NUM_ELEMENTS];
    if (!h_in || !h_out_gpu)
    {
        std::cerr << "Failed to allocate host input/output arrays." << std::endl;
        return 1;
    }

    for (int j = 0; j < NUM_ELEMENTS; j++)
    {
        h_in[j] = dist(rng);
    }

    // --- Device array pointers ---
    uint64_t *d_in = nullptr;  // Use uint64_t for intermediate representation
    uint64_t *d_out = nullptr; // Use uint64_t for intermediate representation

    // --- Intermediate host arrays for conversion ---
    uint64_t *h_input_uint = new uint64_t[NUM_ELEMENTS];
    if (!h_input_uint)
    {
        std::cerr << "Failed to allocate host uint input array." << std::endl;
        return 1;
    }
    uint64_t *h_output_uint = new uint64_t[NUM_ELEMENTS];
    if (!h_output_uint)
    {
        std::cerr << "Failed to allocate host uint output array." << std::endl;
        delete[] h_input_uint;
        return 1;
    }

    // --- Conversion functions (signed int <-> order-preserving unsigned int) ---
    auto hostInt64ToOrderedUInt64 = [](int64_t i) -> uint64_t
    {
        // Flip the MSB of the 64-bit signed integer
        return static_cast<uint64_t>(i) ^ 0x8000000000000000ULL;
    };

    auto hostOrderedUInt64ToInt64 = [](uint64_t ui) -> int64_t
    {
        // Flip the MSB back to restore the original signed integer
        return static_cast<int64_t>(ui ^ 0x8000000000000000ULL);
    };
    // --- End conversion functions ---

    // --- GPU Sorting Process ---
    gpu_timer.Start();

    // 1. Convert the input int64_t array to order-preserving uint64_t
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        h_input_uint[i] = hostInt64ToOrderedUInt64(h_in[i]);
    }

    // 2. Allocate device memory (use sizeof(uint64_t))
    checkCudaErrors(cudaMalloc(&d_in, sizeof(uint64_t) * NUM_ELEMENTS));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(uint64_t) * NUM_ELEMENTS));

    // 3. Copy converted uint64_t data to device (use sizeof(uint64_t))
    checkCudaErrors(cudaMemcpy(d_in, h_input_uint, sizeof(uint64_t) * NUM_ELEMENTS, cudaMemcpyHostToDevice));

    // 4. Perform radix sort on the uint64_t data
    // IMPORTANT: Assumes radix_sort can handle uint64_t* pointers and sorts NUM_ELEMENTS items.
    // If radix_sort was specifically written for uint64_t, it will need modification.
    radix_sort(d_out, d_in, NUM_ELEMENTS);    // Pass uint64_t pointers
    checkCudaErrors(cudaDeviceSynchronize()); // Ensure kernel finishes before stopping timer & copying back

    // 5. Copy sorted uint64_t data back to host (use sizeof(uint64_t))
    checkCudaErrors(cudaMemcpy(h_output_uint, d_out, sizeof(uint64_t) * NUM_ELEMENTS, cudaMemcpyDeviceToHost));

    // 6. Convert the output uint64_t array back to int64_t
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        h_out_gpu[i] = hostOrderedUInt64ToInt64(h_output_uint[i]);
    }

    gpu_timer.Stop();
    // --- End GPU Sorting Process ---

    // --- Cleanup intermediate host arrays ---
    delete[] h_input_uint;
    delete[] h_output_uint;

    // --- Performance Reporting ---
    double gpu_time = (gpu_timer.Elapsed() / 1000.0);
    std::cout << "GPU execution time: " << gpu_time << " seconds\n";

    // --- CPU Verification ---
    std::cout << "Performing CPU sort for verification..." << std::endl;
    std::vector<int64_t> h_in_sorted(h_in, h_in + NUM_ELEMENTS); // Use int64_t vector
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(h_in_sorted.begin(), h_in_sorted.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU execution time: " << cpu_duration.count() << " seconds\n";
    double speedup = cpu_duration.count() / gpu_time;
    std::cout << "Speedup: " << speedup << "x\n";
    std::cout << "Verifying results..." << std::endl;

    bool is_correct = true;
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        if (i < 1000 & i > 900)
        {
            std::cout << "GPU: " << h_out_gpu[i] << ", CPU: " << h_in_sorted[i] << std::endl;
        }
        if (h_out_gpu[i] != h_in_sorted[i])
        {
            std::cerr << "Mismatch at index " << i
                      << ": GPU = " << h_out_gpu[i]
                      << ", CPU = " << h_in_sorted[i] << std::endl;
            is_correct = false;
            break; // Remove or comment out break to see more errors if they exist
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
    // --- End CPU Verification ---

    // --- Final Cleanup ---
    std::cout << "Freeing device memory..." << std::endl;
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_in));
    std::cout << "Freeing host memory..." << std::endl;
    delete[] h_in;
    delete[] h_out_gpu;
    // --- End Final Cleanup ---

    std::cout << "Sort completed." << std::endl;
    return 0;
}