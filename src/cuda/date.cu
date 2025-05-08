#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define MAX 10000000  // Set array size

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

__global__ void findMaxDate(int64_t* input, int64_t* output, int size) {
    __shared__ int64_t warp_maxes[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int64_t local_max = INT64_MIN;
    if (tid < size) local_max = input[tid];

    for (int offset = 16; offset > 0; offset /= 2) {
        int64_t neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = max(local_max, neighbor);
    }

    if (lane_id == 0)
        warp_maxes[warp_id] = local_max;

    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32) {
        local_max = warp_maxes[lane_id];
        for (int offset = 16; offset > 0; offset /= 2) {
            int64_t neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = max(local_max, neighbor);
        }
        if (lane_id == 0)
            atomicMaxInt64(output, local_max);
    }
}

int64_t cpuMaxDate(int64_t* data, int size) {
    int64_t max_val = INT64_MIN;
    for (int i = 0; i < size; i++) {
        if (data[i] > max_val)
            max_val = data[i];
    }
    return max_val;
}

int main() {
    int64_t *h_input = (int64_t*)malloc(MAX * sizeof(int64_t));
    int64_t h_max_cpu, h_max_gpu;
    int64_t *d_input, *d_output;
    
    // Initialize with sample dates (as Unix timestamps)
    for (int i = 0; i < MAX; i++) {
        h_input[i] = 1609459200 + i;  // Starting from 2021-01-01 and incrementing by 1 second
    }

    // CPU computation
    clock_t cpu_start = clock();
    h_max_cpu = cpuMaxDate(h_input, MAX);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU max date: %lld, Time: %.6fs\n", (long long)h_max_cpu, cpu_time);

    // GPU memory allocation and copy
    cudaMalloc(&d_input, MAX * sizeof(int64_t));
    cudaMalloc(&d_output, sizeof(int64_t));
    cudaMemcpy(d_input, h_input, MAX * sizeof(int64_t), cudaMemcpyHostToDevice);
    int64_t min_date = INT64_MIN;
    cudaMemcpy(d_output, &min_date, sizeof(int64_t), cudaMemcpyHostToDevice);

    // Kernel launch
    int blockSize = 256;
    int gridSize = (MAX + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    findMaxDate<<<gridSize, blockSize>>>(d_input, d_output, MAX);
    cudaEventRecord(stop);

    cudaMemcpy(&h_max_gpu, d_output, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU max date: %lld, Time: %.6fs\n", (long long)h_max_gpu, milliseconds / 1000.0);

    // Calculate and print speedup
    printf("Speedup: %.2fx\n", cpu_time / (milliseconds / 1000.0));

    // Cleanup
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}