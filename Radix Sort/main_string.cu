#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

#include "sort.h"
#include "utils.h"
#include "timer.h"

#define NUM_ELEMENTS (1 << 20)

std::mt19937_64 rng(1);
std::uniform_int_distribution<uint64_t> dist;

// Converts up to 8-character ASCII string to uint64_t
uint64_t string_to_uint64(const std::string &str)
{
    uint64_t result = 0;
    for (int i = 0; i < std::min((int)str.length(), 8); ++i)
    {
        result |= (uint64_t)(uint8_t)str[i] << ((7 - i) * 8);
    }
    return result;
}

// Converts uint64_t back to ASCII string
std::string uint64_to_string(uint64_t val)
{
    std::string str;
    for (int i = 0; i < 8; ++i)
    {
        char c = (char)((val >> ((7 - i) * 8)) & 0xFF);
        if (c == 0)
            break;
        str += c;
    }
    return str;
}
__device__ uint64_t string_to_uint64(const char *str)
{
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i)
    {
        char c = str[i];
        if (c == '\0')
            break;
        result |= (uint64_t)(uint8_t)c << ((7 - i) * 8);
    }
    return result;
}

__device__ void uint64_to_string(uint64_t val, char *str_out)
{
    for (int i = 0; i < 8; ++i)
    {
        char c = (char)((val >> ((7 - i) * 8)) & 0xFF);
        if (c == 0)
        {
            str_out[i] = '\0';
            return;
        }
        str_out[i] = c;
    }
    str_out[8] = '\0'; // Null-terminate just in case input has no zero bytes
}

int main()
{
    GpuTimer gpu_timer;
    srand(1);

    std::cout << "NUM_ELEMENTS: " << NUM_ELEMENTS << std::endl;

    std::vector<std::string> strings(NUM_ELEMENTS);
    std::vector<uint64_t> h_in(NUM_ELEMENTS);
    std::vector<std::string> cpu_sorted;

    // Generate random lowercase strings of length 6
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        std::string s;
        for (int j = 0; j < 8; ++j)
        {
            s += charset[rand() % charset.size()];
        }
        strings[i] = s;
        h_in[i] = string_to_uint64(s);
        cpu_sorted.push_back(s);
    }
    uint64_t *h_out_gpu = new uint64_t[NUM_ELEMENTS];

    // Allocate device memory
    uint64_t *d_in = nullptr;
    uint64_t *d_out = nullptr;
    checkCudaErrors(cudaMalloc(&d_in, sizeof(uint64_t) * NUM_ELEMENTS));
    checkCudaErrors(cudaMalloc(&d_out, sizeof(uint64_t) * NUM_ELEMENTS));
    checkCudaErrors(cudaMemcpy(d_in, h_in.data(), sizeof(uint64_t) * NUM_ELEMENTS, cudaMemcpyHostToDevice));

    // Run GPU sort
    gpu_timer.Start();
    radix_sort(d_out, d_in, NUM_ELEMENTS);

    // Copy back sorted data
    checkCudaErrors(cudaMemcpy(h_out_gpu, d_out, sizeof(uint64_t) * NUM_ELEMENTS, cudaMemcpyDeviceToHost));

    // Convert GPU output back to strings
    std::vector<std::string> gpu_sorted(NUM_ELEMENTS);
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        gpu_sorted[i] = uint64_to_string(h_out_gpu[i]);
    }
    gpu_timer.Stop();
    double gpu_time = gpu_timer.Elapsed() / 1000.0;
    std::cout << "GPU execution time: " << gpu_time << " seconds\n";

    // CPU sort
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(cpu_sorted.begin(), cpu_sorted.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU execution time: " << cpu_duration.count() << " seconds\n";
    std::cout << "Speedup: " << cpu_duration.count() / gpu_time << "x\n";
    // Compare
    bool match = true;
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {

        if (cpu_sorted[i] != gpu_sorted[i])
        {

            std::cerr << "Mismatch at index " << i
                      << ": CPU = " << cpu_sorted[i]
                      << ", GPU = " << gpu_sorted[i] << std::endl;
            match = false;
            break;
        }
    }

    if (match)
    {
        std::cout << "GPU radix sort matches std::sort on strings." << std::endl;
    }
    else
    {
        std::cerr << "GPU radix sort does NOT match std::sort on strings." << std::endl;
    }

    // Cleanup
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_in));
    delete[] h_out_gpu;

    std::cout << "String sort completed successfully." << std::endl;
    return 0;
}
