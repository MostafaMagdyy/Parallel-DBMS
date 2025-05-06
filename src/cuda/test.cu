#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <float.h>

#define MAXNUMBER 1000000  // You can change this to control array size

// Enum for different aggregate operations
enum AggregateOperation {
    MIN,
    MAX,
    SUM,
    AVG,
    COUNT
};

// Device function for atomic operations
__device__ float atomicAggregateFloat(float* address, float val, AggregateOperation op) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    switch(op) {
        case MAX:
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed, 
                                __float_as_int(fmaxf(val, __int_as_float(assumed))));
            } while (assumed != old);
            break;
        
        case MIN:
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed, 
                                __float_as_int(fminf(val, __int_as_float(assumed))));
            } while (assumed != old);
            break;
        
        case SUM:
            do {
                assumed = old;
                old = atomicCAS(address_as_int, assumed, 
                                __float_as_int(__int_as_float(assumed) + val));
            } while (assumed != old);
            break;
        
        default:
            break;
    }
    
    return __int_as_float(old);
}

__global__ void aggregateOperation(float* input, float* output, int size, AggregateOperation op) {
    __shared__ float warp_results[32];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32; 
    int warp_id = threadIdx.x / 32; 
    
    float local_result;
    
    // Initialize based on operation
    switch(op) {
        case MAX:
            local_result = -FLT_MAX;
            break;
        case MIN:
            local_result = FLT_MAX;
            break;
        case SUM:
        case COUNT:
            local_result = 0;
            break;
        default:
            local_result = 0;
    }
    
    // Gather local results
    if (tid < size) {
        switch(op) {
            case MAX:
                local_result = fmaxf(local_result, input[tid]);
                break;
            case MIN:
                local_result = fminf(local_result, input[tid]);
                break;
            case SUM:
                local_result += input[tid];
                break;
            case COUNT:
                local_result += 1;
                break;
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        float neighbor;
        switch(op) {
            case MAX:
                neighbor = __shfl_down_sync(0xffffffff, local_result, offset);
                local_result = fmaxf(local_result, neighbor);
                break;
            case MIN:
                neighbor = __shfl_down_sync(0xffffffff, local_result, offset);
                local_result = fminf(local_result, neighbor);
                break;
            case SUM:
            case COUNT:
                neighbor = __shfl_down_sync(0xffffffff, local_result, offset);
                local_result += neighbor;
                break;
        }
    }

    // Store warp-level results
    if (lane_id == 0) {
        warp_results[warp_id] = local_result;
    }

    __syncthreads();

    // Block-level reduction
    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32) {
        local_result = warp_results[lane_id];

        for (int offset = 16; offset > 0; offset /= 2) {
            float neighbor;
            switch(op) {
                case MAX:
                    neighbor = __shfl_down_sync(0xffffffff, local_result, offset);
                    local_result = fmaxf(local_result, neighbor);
                    break;
                case MIN:
                    neighbor = __shfl_down_sync(0xffffffff, local_result, offset);
                    local_result = fminf(local_result, neighbor);
                    break;
                case SUM:
                case COUNT:
                    neighbor = __shfl_down_sync(0xffffffff, local_result, offset);
                    local_result += neighbor;
                    break;
            }
        }

        // Atomic update to output
        if (lane_id == 0) {
            switch(op) {
                case MAX:
                case MIN:
                case SUM:
                    atomicAggregateFloat(output, local_result, op);
                    break;
                case COUNT:
                    atomicAdd(output, local_result);
                    break;
            }
        }
    }
}

// CPU reduction functions
float reduceCPU(float *data, int size, AggregateOperation op) {
    float result;
    switch(op) {
        case MAX:
            result = -FLT_MAX;
            for (int i = 0; i < size; ++i)
                result = fmaxf(result, data[i]);
            break;
        case MIN:
            result = FLT_MAX;
            for (int i = 0; i < size; ++i)
                result = fminf(result, data[i]);
            break;
        case SUM:
            result = 0;
            for (int i = 0; i < size; ++i)
                result += data[i];
            break;
        case COUNT:
            result = size;
            break;
        default:
            result = 0;
    }
    return result;
}

int main() {
    int size = MAXNUMBER;
    float *h_input = (float *)malloc(size * sizeof(float));

    // Initialize input array with varied values
    for (int i = 0; i < size; ++i) {
        h_input[i] = i+1.0f;
    }

    // Operations to test
    AggregateOperation ops[] = {MIN, MAX, SUM, COUNT};
    const char* op_names[] = {"Min", "Max", "Sum", "Count"};

    for (auto op : ops) {
        float *d_input, *d_output;
        cudaMalloc(&d_input, size * sizeof(float));
        cudaMalloc(&d_output, sizeof(float));
        
        // Initialize device memory
        cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
        
        float init_val = 0;
        switch(op) {
            case MAX:
                init_val = -FLT_MAX;
                break;
            case MIN:
                init_val = FLT_MAX;
                break;
        }
        cudaMemset(d_output, init_val, sizeof(float));

        // GPU Timing
        auto gpu_start = std::chrono::high_resolution_clock::now();
        int blockSize = 256;
        int gridSize = (size + blockSize - 1) / blockSize;
        aggregateOperation<<<gridSize, blockSize>>>(d_input, d_output, size, op);
        cudaDeviceSynchronize();
        auto gpu_end = std::chrono::high_resolution_clock::now();

        // Retrieve GPU result
        float h_output;
        cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        // CPU Timing
        auto cpu_start = std::chrono::high_resolution_clock::now();
        float cpu_result = reduceCPU(h_input, size, op);
        auto cpu_end = std::chrono::high_resolution_clock::now();

        // Timing and output
        std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
        std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

        printf("%s Operation:\n", op_names[op]);
        printf("GPU result: %.3f in %.3f ms\n", h_output, gpu_time.count());
        printf("CPU result: %.3f in %.3f ms\n", cpu_result, cpu_time.count());

        double speedup = cpu_time.count() / gpu_time.count();
        printf("Speedup: %.2fx\n\n", speedup);

        // Clean up
        cudaFree(d_input);
        cudaFree(d_output);
    }

    free(h_input);
    return 0;
}