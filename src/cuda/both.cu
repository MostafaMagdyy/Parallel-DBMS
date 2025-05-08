#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <climits>

#define MAX 10000000  // Set array size

// Atomic max for floats
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, 
                       __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}
__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}
__device__ float atomicAddFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(__int_as_float(assumed) + val));
    } while (assumed != old);
    
    return __int_as_float(old);
}

__device__ int64_t atomicMaxInt64(int64_t* address, int64_t val) {
    int64_t old = *address, assumed;
    
    do {
        assumed = old;
        old = atomicCAS((unsigned long long*)address, 
                       (unsigned long long)assumed, 
                       (unsigned long long)max(val, assumed));
    } while (assumed != old);
    
    return old;
}
__device__ int64_t atomicMinInt64(int64_t* address, int64_t val) {
    int64_t old = *address, assumed;
    
    do {
        assumed = old;
        old = atomicCAS((unsigned long long*)address, 
                       (unsigned long long)assumed, 
                       (unsigned long long)min(val, assumed));
    } while (assumed != old);
    
    return old;
}
// Template kernel for finding maximum value
template <typename T>
__global__ void findMaxElement(T* input, T* output, int size, bool isFloat) {
    __shared__ T warp_maxes[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Initialize based on type
    T local_max = isFloat ? -INFINITY : INT64_MIN;
    if (tid < size) local_max = input[tid];

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        T neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        if (isFloat) {
            local_max = fmaxf(local_max, neighbor);
        } else {
            local_max = max(local_max, neighbor);
        }
    }

    if (lane_id == 0) {
        warp_maxes[warp_id] = local_max;
    }

    __syncthreads();

    // Final reduction for first warp
    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32) {
        local_max = warp_maxes[lane_id];
        
        for (int offset = 16; offset > 0; offset /= 2) {
            T neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
            if (isFloat) {
                local_max = fmaxf(local_max, neighbor);
            } else {
                local_max = max(local_max, neighbor);
            }
        }
        
        if (lane_id == 0) {
            if (isFloat) {
                atomicMaxFloat((float*)output, local_max);
            } else {
                atomicMaxInt64((int64_t*)output, local_max);
            }
        }
    }
}

// CPU functions
template <typename T>
T cpuMax(T* data, int size, bool isFloat) {
    T max_val = isFloat ? -INFINITY : INT64_MIN;
    for (int i = 0; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

int main() {
    // Test float data
    float *h_float_input = (float*)malloc(MAX * sizeof(float));
    float h_float_max_cpu, h_float_max_gpu;
    float *d_float_input, *d_float_output;
    
    for (int i = 0; i < MAX; i++) {
        h_float_input[i] = i+1;  // Simple increasing values
    }

    // Test int64_t data (dates)
    int64_t *h_date_input = (int64_t*)malloc(MAX * sizeof(int64_t));
    int64_t h_date_max_cpu, h_date_max_gpu;
    int64_t *d_date_input, *d_date_output;
    
    for (int i = 0; i < MAX; i++) {
        h_date_input[i] = 1609459200LL + i;  // Unix timestamps starting from 2021-01-01
    }

    // GPU memory allocations
    cudaMalloc(&d_float_input, MAX * sizeof(float));
    cudaMalloc(&d_float_output, sizeof(float));
    cudaMalloc(&d_date_input, MAX * sizeof(int64_t));
    cudaMalloc(&d_date_output, sizeof(int64_t));

    // Copy data to GPU
    cudaMemcpy(d_float_input, h_float_input, MAX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_date_input, h_date_input, MAX * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Initialize outputs
    float init_float = -INFINITY;
    int64_t init_date = INT64_MIN;
    cudaMemcpy(d_float_output, &init_float, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_date_output, &init_date, sizeof(int64_t), cudaMemcpyHostToDevice);

    // Kernel configuration
    int blockSize = 256;
    int gridSize = (MAX + blockSize - 1) / blockSize;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Process float data
    // CPU computation
    clock_t cpu_start = clock();
    h_float_max_cpu = cpuMax(h_float_input, MAX, true);
    clock_t cpu_end = clock();
    double cpu_time_float = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU float max: %.3f, Time: %.6fs\n", h_float_max_cpu, cpu_time_float);

    // GPU computation
    cudaEventRecord(start);
    findMaxElement<float><<<gridSize, blockSize>>>(d_float_input, d_float_output, MAX, true);
    cudaEventRecord(stop);
    cudaMemcpy(&h_float_max_gpu, d_float_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds_float = 0;
    cudaEventElapsedTime(&milliseconds_float, start, stop);
    printf("GPU float max: %.3f, Time: %.6fs\n", h_float_max_gpu, milliseconds_float / 1000.0);
    printf("Float Speedup: %.2fx\n", cpu_time_float / (milliseconds_float / 1000.0));

    // Process date data
    // CPU computation
    cpu_start = clock();
    h_date_max_cpu = cpuMax(h_date_input, MAX, false);
    cpu_end = clock();
    double cpu_time_date = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU date max: %lld, Time: %.6fs\n", (long long)h_date_max_cpu, cpu_time_date);

    // GPU computation
    cudaEventRecord(start);
    findMaxElement<int64_t><<<gridSize, blockSize>>>(d_date_input, d_date_output, MAX, false);
    cudaEventRecord(stop);
    cudaMemcpy(&h_date_max_gpu, d_date_output, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds_date = 0;
    cudaEventElapsedTime(&milliseconds_date, start, stop);
    printf("GPU date max: %lld, Time: %.6fs\n", (long long)h_date_max_gpu, milliseconds_date / 1000.0);
    printf("Date Speedup: %.2fx\n", cpu_time_date / (milliseconds_date / 1000.0));

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