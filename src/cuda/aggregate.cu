#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <climits>
#include <float.h>
#include "../headers/device_struct.h"
#include "../operators/operator_enums.h"
#include "../headers/enums.h"
#define MAXX 10000000
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

__device__ int64_t atomicMaxInt64(int64_t *address, int64_t val)
{
    int64_t old = *address, assumed;

    do
    {
        assumed = old;
        old = atomicCAS((unsigned long long *)address,
                        (unsigned long long)assumed,
                        (unsigned long long)max(val, assumed));
    } while (assumed != old);

    return old;
}
__device__ int64_t atomicMinInt64(int64_t *address, int64_t val)
{
    int64_t old = *address, assumed;

    do
    {
        assumed = old;
        old = atomicCAS((unsigned long long *)address,
                        (unsigned long long)assumed,
                        (unsigned long long)min(val, assumed));
    } while (assumed != old);

    return old;
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
    T local_max = (dtype == ColumnType::FLOAT) ? -FLT_MAX : INT64_MIN;

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
                atomicMaxInt64((int64_t *)output, local_max);
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
        local_min = (lane_id < (blockDim.x + 31) / 32) ? warp_min[lane_id] : (dtype == ColumnType::FLOAT) ? FLT_MAX
                                                                                                          : LLONG_MAX;

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
                atomicMinInt64((int64_t *)output, local_min);
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
int64_t cpuMaxDate(int64_t *data, int size)
{
    int64_t max_val = INT64_MIN;
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

int64_t cpuMinDate(int64_t *arr, int n)
{
    int64_t min_val = INT64_MAX;
    for (int i = 0; i < n; ++i)
    {
        if (arr[i] < min_val)
            min_val = arr[i];
    }
    return min_val;
}

void tobecalledfromCPU(DeviceStruct *devicestructarr, AggregateFunctionType *arr, int opnums, void **result)
{
    std::vector<void *> value_ptrs(opnums);
    for (int i = 0; i < opnums; i++)
    {
        cudaMalloc(&value_ptrs[i], sizeof(result[i]));
        // set initial values
        switch (devicestructarr[i].type)
        {
        case ColumnType::FLOAT:
            switch (arr[i])
            {
            case AggregateFunctionType::SUM:
            {
                float zero = 0.0f;
                cudaMemcpy(value_ptrs[i], &zero, sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case AggregateFunctionType::MAX:
            {
                float neg_inf = -FLT_MAX;
                cudaMemcpy(value_ptrs[i], &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            case AggregateFunctionType::MIN:
            {
                float pos_inf = FLT_MAX;
                cudaMemcpy(value_ptrs[i], &pos_inf, sizeof(float), cudaMemcpyHostToDevice);
                break;
            }
            }
            break;
        case ColumnType::DATE:
            switch (arr[i])
            {
            case AggregateFunctionType::SUM:
                throw "SUM operation not supported for date column";
                break;
            case AggregateFunctionType::MAX:
            {
                int64_t min_int64 = INT64_MIN;
                cudaMemcpy(value_ptrs[i], &min_int64, sizeof(int64_t), cudaMemcpyHostToDevice);
                break;
            }
            case AggregateFunctionType::MIN:
            {
                int64_t max_int64 = INT64_MAX;
                cudaMemcpy(value_ptrs[i], &max_int64, sizeof(int64_t), cudaMemcpyHostToDevice);
                break;
            }
            }
            break;
        default:
            printf("Unsupported data type for aggregate function\n");
            break;
        }
    }
    for (int i = 0; i < opnums; i++)
    {
        int blockSize = 256;
        int gridSize = (devicestructarr[i].numRows + blockSize - 1) / blockSize;
        if (arr[i] == AggregateFunctionType::SUM)
        {
            if (devicestructarr[i].type == ColumnType::FLOAT)
            {
                sumKernel<<<gridSize, blockSize>>>((float *)devicestructarr[i].device_ptr, (float *)value_ptrs[i], devicestructarr[i].numRows);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
                }
                else
                {
                    printf("Kernel launch succeeded\n");
                }
            }
            else
            {
                printf("Unsupported data type for SUM operation\n");
            }
        }
        else if (arr[i] == AggregateFunctionType::MAX)
        {
            if (devicestructarr[i].type == ColumnType::FLOAT)
            {
                maxKernel<float><<<gridSize, blockSize>>>((float *)devicestructarr[i].device_ptr, (float *)value_ptrs[i], devicestructarr[i].numRows, devicestructarr[i].type);
            }
            else if (devicestructarr[i].type == ColumnType::DATE)
            {
                maxKernel<int64_t><<<gridSize, blockSize>>>((int64_t *)devicestructarr[i].device_ptr, (int64_t *)value_ptrs[i], devicestructarr[i].numRows, devicestructarr[i].type);
            }
            else
            {
                printf("Unsupported data type for MAX operation\n");
            }
        }
        else if (arr[i] == AggregateFunctionType::MIN)
        {
            if (devicestructarr[i].type == ColumnType::FLOAT)
            {
                minKernel<float><<<gridSize, blockSize>>>((float *)devicestructarr[i].device_ptr, (float *)value_ptrs[i], devicestructarr[i].numRows, devicestructarr[i].type);
            }
            else if (devicestructarr[i].type == ColumnType::DATE)
            {
                minKernel<int64_t><<<gridSize, blockSize>>>((int64_t *)devicestructarr[i].device_ptr, (int64_t *)value_ptrs[i], devicestructarr[i].numRows, devicestructarr[i].type);
            }
            else
            {
                printf("Unsupported data type for MIN operation\n");
            }
        }
        else
        {
            printf("Unsupported operation\n");
        }
    }
    for (int i = 0; i < opnums; i++)
    {
        cudaMemcpy(result[i], value_ptrs[i], sizeof(result[i]), cudaMemcpyDeviceToHost);
    }
}
// int main()
// {
//     // Test float data
//     float *h_float_input = (float *)malloc(MAXX * sizeof(float));
//     float h_float_max_cpu, h_float_max_gpu;
//     float *d_float_input, *d_float_output;

//     for (int i = 0; i < MAXX; i++)
//     {
//         h_float_input[i] = i + 1; // Simple increasing values
//     }

//     // Test int64_t data (dates)
//     int64_t *h_date_input = (int64_t *)malloc(MAXX * sizeof(int64_t));
//     int64_t h_date_max_cpu, h_date_max_gpu;
//     int64_t *d_date_input, *d_date_output;

//     for (int i = 0; i < MAXX; i++)
//     {
//         h_date_input[i] = 1609459200LL + i; // Unix timestamps starting from 2021-01-01
//     }

//     // GPU memory allocations
//     cudaMalloc(&d_float_input, MAXX * sizeof(float));
//     cudaMalloc(&d_float_output, sizeof(float));
//     cudaMalloc(&d_date_input, MAXX * sizeof(int64_t));
//     cudaMalloc(&d_date_output, sizeof(int64_t));

//     // Copy data to GPU
//     cudaMemcpy(d_float_input, h_float_input, MAXX * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_date_input, h_date_input, MAXX * sizeof(int64_t), cudaMemcpyHostToDevice);

//     // Initialize outputs
//     float init_float = INFINITY;
//     int64_t init_date = INT64_MAX;
//     cudaMemcpy(d_float_output, &init_float, sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_date_output, &init_date, sizeof(int64_t), cudaMemcpyHostToDevice);

//     // Kernel configuration
//     int blockSize = 256;
//     int gridSize = (MAXX + blockSize - 1) / blockSize;

//     // CUDA events for timing
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // Process float data
//     // CPU computation
//     clock_t cpu_start = clock();
//     h_float_max_cpu = cpuMinFloat(h_float_input, MAXX);
//     clock_t cpu_end = clock();
//     double cpu_time_float = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
//     printf("CPU float min: %.3f, Time: %.6fs\n", h_float_max_cpu, cpu_time_float);

//     // GPU computation
//     cudaEventRecord(start);
//     minKernel<float><<<gridSize, blockSize>>>(d_float_input, d_float_output, MAXX, ColumnType::FLOAT);
//     cudaEventRecord(stop);
//     cudaMemcpy(&h_float_max_gpu, d_float_output, sizeof(float), cudaMemcpyDeviceToHost);
//     cudaEventSynchronize(stop);

//     float milliseconds_float = 0;
//     cudaEventElapsedTime(&milliseconds_float, start, stop);
//     printf("GPU float min: %.3f, Time: %.6fs\n", h_float_max_gpu, milliseconds_float / 1000.0);
//     printf("Float Speedup: %.2fx\n", cpu_time_float / (milliseconds_float / 1000.0));

//     // Process date data
//     // CPU computation
//     cpu_start = clock();
//     h_date_max_cpu = cpuMinDate(h_date_input, MAXX); // <== FIXED
//     cpu_end = clock();
//     double cpu_time_date = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
//     printf("CPU date max: %lld, Time: %.6fs\n", (long long)h_date_max_cpu, cpu_time_date);

//     // GPU computation
//     cudaEventRecord(start);
//     minKernel<int64_t><<<gridSize, blockSize>>>(d_date_input, d_date_output, MAXX, ColumnType::DATE);
//     cudaEventRecord(stop);
//     cudaMemcpy(&h_date_max_gpu, d_date_output, sizeof(int64_t), cudaMemcpyDeviceToHost);
//     cudaEventSynchronize(stop);

//     float milliseconds_date = 0;
//     cudaEventElapsedTime(&milliseconds_date, start, stop);
//     printf("GPU date max: %lld, Time: %.6fs\n", (long long)h_date_max_gpu, milliseconds_date / 1000.0);
//     printf("Date Speedup: %.2fx\n", cpu_time_date / (milliseconds_date / 1000.0));

//     // Cleanup
//     free(h_float_input);
//     free(h_date_input);
//     cudaFree(d_float_input);
//     cudaFree(d_float_output);
//     cudaFree(d_date_input);
//     cudaFree(d_date_output);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     return 0;
// }