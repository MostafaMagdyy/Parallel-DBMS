#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

#define MAXNUMBER 1000000  // You can change this to control array size

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, 
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}

__global__ void findMaxElement(float* input, float* output, int size) {
    __shared__ float warp_maxes[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32; 
    int warp_id = threadIdx.x / 32; 
    
    float local_max = -INFINITY;
    if (tid < size) {
        local_max = input[tid];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        float neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, neighbor);
    }

    if (lane_id == 0) {
        warp_maxes[warp_id] = local_max;
    }

    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32) {
        local_max = warp_maxes[lane_id];

        for (int offset = 16; offset > 0; offset /= 2) {
            float neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, neighbor);
        }

        if (lane_id == 0) {
            atomicMaxFloat(output, local_max);
        }
    }
}

float findMaxCPU(float *data, int size) {
    float maxVal = -INFINITY;
    for (int i = 0; i < size; ++i) {
        if (data[i] > maxVal)
            maxVal = data[i];
    }
    return maxVal;
}

int main() {
    int size = MAXNUMBER;
    float *h_input = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        h_input[i] = (float)(i + 1);  // Initialize from 1 to MAXNUMBER
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    float neg_inf = -INFINITY;
    cudaMemcpy(d_output, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Timing GPU
    findMaxElement<<<gridSize, blockSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();

    float h_output;
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Timing CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = findMaxCPU(h_input, size);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    printf("GPU result: %.3f in %.3f ms\n", h_output, gpu_time.count());
    printf("CPU result: %.3f in %.3f ms\n", cpu_result, cpu_time.count());

    double speedup = cpu_time.count() / gpu_time.count();
    printf("Speedup: %.2fx\n", speedup);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
