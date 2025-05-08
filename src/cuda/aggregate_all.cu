#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <climits>
#include <float.h>
#include "../operators/operator_enums.h"
#include "../headers/enums.h"
#define MAXX 4
// Atomic max for floats
__device__ float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}
__device__ float atomicMinFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}
__device__ float atomicAddFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int;
    int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(__int_as_float(assumed) + val));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ long long atomicMaxLongLong(long long *address, long long val)
{
    long long old = *address, assumed;

    do
    {
        assumed = old;
        old = atomicCAS((unsigned long long *)address,
                        (unsigned long long)assumed,
                        (unsigned long long)max(val, assumed));
    } while (assumed != old);

    return old;
}

__device__ long long atomicMinLongLong(long long* address, long long val)
{
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                       (unsigned long long)min((long long)assumed, val));
    } while (assumed != old);

    return (long long)old;
}

// Template kernel for finding maximum value
template <typename T>
__global__ void maxKernel(T *input, T *output, int size, ColumnType dtype)
{
    __shared__ T warp_max[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Initialize based on data type
    T local_max = (dtype == ColumnType::FLOAT) ? -FLT_MAX : LLONG_MIN;

    if (tid < size)
    {
        local_max = input[tid];
    }

    // Warp-level reduction with explicit comparison
    for (int offset = 16; offset > 0; offset /= 2)
    {
        T neighbor = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (neighbor > local_max)
            local_max = neighbor;
    }

    if (lane_id == 0)
    {
        warp_max[warp_id] = local_max;
    }

    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32)
    {
        local_max = warp_max[lane_id];

        for (int offset = 16; offset > 0; offset /= 2)
        {
            T neighbor = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            if (neighbor > local_max)
                local_max = neighbor;
        }

        if (lane_id == 0)
        {
            if (dtype == ColumnType::FLOAT)
            {
                atomicMaxFloat((float *)output, local_max);
            }
            else
            {
                // Use built-in atomicMax for long long values
                atomicMax((unsigned long long *)output, (unsigned long long)local_max);
            }
        }
    }
}

template <typename T>
__global__ void minKernel(T *input, T *output, int size, ColumnType dtype)
{
    __shared__ T warp_min[32]; // One per warp

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    T local_min = (dtype == ColumnType::FLOAT) ? FLT_MAX : LLONG_MAX;

    if (tid < size)
    {
        local_min = input[tid];
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
    {
        T neighbor = __shfl_down_sync(0xffffffff, local_min, offset);
        local_min = min(local_min, neighbor);
    }
    
    if (lane_id == 0)
    {
        warp_min[warp_id] = local_min;
    }
    
    __syncthreads();

    if (warp_id == 0)
    {
        local_min = (lane_id < (blockDim.x + 31) / 32) ? warp_min[lane_id] : (dtype == ColumnType::FLOAT ? FLT_MAX : LLONG_MAX);

        for (int offset = 16; offset > 0; offset /= 2)
        {
            T neighbor = __shfl_down_sync(0xffffffff, local_min, offset);
            local_min = min(local_min, neighbor);
        }

        if (lane_id == 0)
        {
            if (dtype == ColumnType::FLOAT)
            {
                atomicMinFloat((float *)output, local_min);
            }
            else
            {
                atomicMin((unsigned long long *)output, (unsigned long long)local_min);
            }
        }
    }
}

__global__ void sumKernel(float *input, float *output, int size)
{
    __shared__ float warp_sum[32]; // One per warp

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float local_sum = 0.0f;

    if (tid < size)
    {
        local_sum = input[tid];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float neighbor = __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum += neighbor;
    }

    if (lane_id == 0)
    {
        warp_sum[warp_id] = local_sum;
    }

    __syncthreads();

    if (warp_id == 0)
    {
        local_sum = (lane_id < (blockDim.x + 31) / 32) ? warp_sum[lane_id] : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2)
        {
            float neighbor = __shfl_down_sync(0xffffffff, local_sum, offset);
            local_sum += neighbor;
        }

        if (lane_id == 0)
        {
            atomicAddFloat(output, local_sum);
        }
    }
}

float cpuMaxFloat(const float *data, int size)
{
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++)
    {
        if (data[i] > max_val)
        {
            max_val = data[i];
        }
    }
    return max_val;
}

long long cpuMaxDate(long long *data, int size)
{
    long long max_val = LLONG_MIN;
    for (int i = 0; i < size; i++)
    {
        if (data[i] > max_val)
            max_val = data[i];
    }
    return max_val;
}

float cpuMinFloat(float *arr, int n)
{
    float min_val = INFINITY;
    for (int i = 0; i < n; ++i)
    {
        if (arr[i] < min_val)
            min_val = arr[i];
    }
    return min_val;
}

float cpuSumFloat(float *arr, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
        sum += arr[i];
    return sum;
}

long long cpuMinDate(long long *arr, int n)
{
    long long min_val = LLONG_MAX;
    for (int i = 0; i < n; ++i)
    {
        if (arr[i] < min_val)
            min_val = arr[i];
    }
    return min_val;
}

int main()
{
    // Test float data
    float *h_float_input = (float *)malloc(MAXX * sizeof(float));
    float h_float_max_cpu, h_float_max_gpu;
    float *d_float_input, *d_float_output;

    for (int i = 0; i < MAXX; i++)
    {
        h_float_input[i] = 5+i; // Simple increasing values
    }

    // Test long long data (dates)
    long long *h_date_input = (long long *)malloc(MAXX * sizeof(long long));
    long long h_date_max_cpu, h_date_max_gpu;
    long long *d_date_input, *d_date_output;
    h_date_input[0] = 1640296800;
    h_date_input[1] = 1623880800;
    h_date_input[2] = 1668636000;
    h_date_input[3] = 169014960000;

    // GPU memory allocations
    cudaMalloc(&d_float_input, MAXX * sizeof(float));
    cudaMalloc(&d_float_output, sizeof(float));
    cudaMalloc(&d_date_input, MAXX * sizeof(long long));
    cudaMalloc(&d_date_output, sizeof(long long));

    // Copy data to GPU
    cudaMemcpy(d_float_input, h_float_input, MAXX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_date_input, h_date_input, MAXX * sizeof(long long), cudaMemcpyHostToDevice);

    // Initialize outputs
    float init_float = INFINITY;
    unsigned long long init_date = ULLONG_MAX;
    cudaMemcpy(d_float_output, &init_float, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_date_output, &init_date, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Kernel configuration
    int blockSize = 256;
    int gridSize = (MAXX + blockSize - 1) / blockSize;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Process float data
    // CPU computation
    clock_t cpu_start = clock();
    // h_float_max_cpu = cpuMinFloat(h_float_input, MAXX);
    // clock_t cpu_end = clock();
    // double cpu_time_float = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    // printf("CPU float min: %.3f, Time: %.6fs\n", h_float_max_cpu, cpu_time_float);

    // // GPU computation
    // cudaEventRecord(start);
    // minKernel<float><<<gridSize, blockSize>>>(d_float_input, d_float_output, MAXX, ColumnType::FLOAT);
    // cudaEventRecord(stop);
    // cudaMemcpy(&h_float_max_gpu, d_float_output, sizeof(float), cudaMemcpyDeviceToHost);
    // cudaEventSynchronize(stop);

    // float milliseconds_float = 0;
    // cudaEventElapsedTime(&milliseconds_float, start, stop);
    // printf("GPU float min: %.3f, Time: %.6fs\n", h_float_max_gpu, milliseconds_float / 1000.0);
    // printf("Float Speedup: %.2fx\n", cpu_time_float / (milliseconds_float / 1000.0));

    // Process date data
    // CPU computation
    // cpu_start = clock();
    h_date_max_cpu = cpuMinDate(h_date_input, MAXX); // <== FIXED
    // cpu_end = clock();
    // double cpu_time_date = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    // printf("CPU date min: %lld, Time: %.6fs\n", (long long)h_date_max_cpu, cpu_time_date);

    // GPU computation
    cudaEventRecord(start);
    minKernel<long long><<<gridSize, blockSize>>>(d_date_input, d_date_output, MAXX, ColumnType::DATE);
    cudaEventRecord(stop);
    cudaMemcpy(&h_date_max_gpu, d_date_output, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds_date = 0;
    cudaEventElapsedTime(&milliseconds_date, start, stop);
    printf("CPU date min: %lld\n", h_date_max_cpu);
    printf("GPU date min: %lld, Time: %.6fs\n", h_date_max_gpu, milliseconds_date / 1000.0);
    // printf("Date Speedup: %.2fx\n", cpu_time_date / (milliseconds_date / 1000.0));

    // Cleanup
    free(h_float_input);
    free(h_date_input);
    cudaFree(d_float_input);
    cudaFree(d_float_output);
    cudaFree(d_date_input);
    cudaFree(d_date_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}