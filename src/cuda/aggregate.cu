#include "aggregate.cuh"
#include <stdio.h>
#include <math.h>
#include <chrono>

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

__device__ int64_t atomicAddInt64(int64_t* address, int64_t val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, (unsigned long long int)(val + assumed));
    } while (assumed != old);
    
    return (int64_t)old;
}

__device__ int64_t atomicMaxInt64(int64_t* address, int64_t val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                     (unsigned long long int)max((int64_t)val, (int64_t)assumed));
    } while (assumed != old);
    
    return (int64_t)old;
}

__device__ int64_t atomicMinInt64(int64_t* address, int64_t val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                     (unsigned long long int)min((int64_t)val, (int64_t)assumed));
    } while (assumed != old);
    
    return (int64_t)old;
}

// Warp reduction for float values
__device__ void warpReduce(float& value, int aggregation_type) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float neighbor = __shfl_down_sync(0xffffffff, value, offset);
        
        switch (aggregation_type) {
            case AGG_MAX: value = fmaxf(value, neighbor); break;
            case AGG_MIN: value = fminf(value, neighbor); break;
            case AGG_SUM:
            case AGG_AVG: value += neighbor; break;
            case AGG_COUNT: value += neighbor; break;
        }
    }
}

// Warp reduction for int64_t values
__device__ void warpReduce(int64_t& value, int aggregation_type) {
    for (int offset = 16; offset > 0; offset /= 2) {
        int64_t neighbor = __shfl_down_sync(0xffffffff, value, offset);
        
        switch (aggregation_type) {
            case AGG_MAX: value = max(value, neighbor); break;
            case AGG_MIN: value = min(value, neighbor); break;
            case AGG_SUM:
            case AGG_AVG: value += neighbor; break;
            case AGG_COUNT: value += neighbor; break;
        }
    }
}

__global__ void aggregateKernel(void* input, AggregateResult* output, 
                             int size, int aggregation_type, int value_type) {
    __shared__ AggregateResult warp_results[32];  // For 32 warps per block max
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32; 
    int warp_id = threadIdx.x / 32;
    
    AggregateResult local_result;
    
    switch (aggregation_type) {
        case AGG_MIN:
            if (value_type == TYPE_FLOAT) local_result.float_val = INFINITY;
            else local_result.int_val = INT64_MAX;
            break;
        case AGG_MAX:
            if (value_type == TYPE_FLOAT) local_result.float_val = -INFINITY;
            else local_result.int_val = INT64_MIN;
            break;
        case AGG_SUM:
        case AGG_AVG:
            if (value_type == TYPE_FLOAT) local_result.float_val = 0.0f;
            else local_result.int_val = 0;
            break;
        case AGG_COUNT:
            local_result.int_val = 0;
            break;
    }
    local_result.count = 0;
    
    if (tid < size) {
        if (value_type == TYPE_FLOAT) {
            float val = ((float*)input)[tid];
            
            switch (aggregation_type) {
                case AGG_MIN: local_result.float_val = val; break;
                case AGG_MAX: local_result.float_val = val; break;
                case AGG_SUM: 
                case AGG_AVG: local_result.float_val = val; break;
                case AGG_COUNT: local_result.int_val = 1; break;
            }
        } 
        else {  // TYPE_INT or TYPE_DATE
            int64_t val = ((int64_t*)input)[tid];
            
            switch (aggregation_type) {
                case AGG_MIN: local_result.int_val = val; break;
                case AGG_MAX: local_result.int_val = val; break;
                case AGG_SUM:
                case AGG_AVG: local_result.int_val = val; break;
                case AGG_COUNT: local_result.int_val = 1; break;
            }
        }
        
        // For AVG, we need to keep track of the count
        if (aggregation_type == AGG_AVG) {
            local_result.count = 1;
        }
    }
    
    // Warp-level reduction
    if (value_type == TYPE_FLOAT) {
        warpReduce(local_result.float_val, aggregation_type);
    } else {
        warpReduce(local_result.int_val, aggregation_type);
    }
    
    // Count reduction for AVG
    if (aggregation_type == AGG_AVG) {
        warpReduce(local_result.count, AGG_SUM);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = local_result;
    }
    
    __syncthreads();
    
    // Block-level reduction (only first warp)
    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32) {
        local_result = warp_results[lane_id];
        
        // Reduce across warps
        if (value_type == TYPE_FLOAT) {
            warpReduce(local_result.float_val, aggregation_type);
        } else {
            warpReduce(local_result.int_val, aggregation_type);
        }
        
        // Count reduction for AVG
        if (aggregation_type == AGG_AVG) {
            warpReduce(local_result.count, AGG_SUM);
        }
        
        // First thread updates global result using atomics
        if (lane_id == 0) {
            if (value_type == TYPE_FLOAT) {
                switch (aggregation_type) {
                    case AGG_MIN:
                        atomicMinFloat(&output->float_val, local_result.float_val);
                        break;
                    case AGG_MAX:
                        atomicMaxFloat(&output->float_val, local_result.float_val);
                        break;
                    case AGG_SUM:
                    case AGG_AVG:
                        atomicAdd(&output->float_val, local_result.float_val);
                        break;
                    case AGG_COUNT:
                        atomicAdd((unsigned long long int*)&output->int_val, local_result.int_val);
                        break;
                }
            } else {  // TYPE_INT or TYPE_DATE
                switch (aggregation_type) {
                    case AGG_MIN:
                        atomicMinInt64(&output->int_val, local_result.int_val);
                        break;
                    case AGG_MAX:
                        atomicMaxInt64(&output->int_val, local_result.int_val);
                        break;
                    case AGG_SUM:
                    case AGG_AVG:
                        atomicAddInt64(&output->int_val, local_result.int_val);
                        break;
                    case AGG_COUNT:
                        atomicAddInt64(&output->int_val, local_result.int_val);
                        break;
                }
            }
            
            // For AVG, we need to update the count
            if (aggregation_type == AGG_AVG) {
                atomicAddInt64(&output->count, local_result.count);
            }
        }
    }
}

void computeAggregate(void* data, int size, AggregateType agg_type, 
                     ValueType val_type, AggregateResult& result) {
    void* d_input;
    AggregateResult* d_output;
    fprintf(stderr, "computeAggregate on GPU called:\n");    
    size_t data_size = (val_type == TYPE_FLOAT) ? sizeof(float) : sizeof(int64_t);
    cudaMalloc(&d_input, size * data_size);
    cudaMalloc(&d_output, sizeof(AggregateResult));    
    cudaMemcpy(d_input, data, size * data_size, cudaMemcpyHostToDevice);
    
    AggregateResult init_result;
    switch (agg_type) {
        case AGG_MIN:
            if (val_type == TYPE_FLOAT) init_result.float_val = INFINITY;
            else init_result.int_val = INT64_MAX;
            break;
        case AGG_MAX:
            if (val_type == TYPE_FLOAT) init_result.float_val = -INFINITY;
            else init_result.int_val = INT64_MIN;
            break;
        case AGG_SUM:
        case AGG_AVG:
        case AGG_COUNT:
            if (val_type == TYPE_FLOAT) init_result.float_val = 0.0f;
            else init_result.int_val = 0;
            break;
    }
    init_result.count = 0;
    
    cudaMemcpy(d_output, &init_result, sizeof(AggregateResult), cudaMemcpyHostToDevice);
    
    // Configure and launch the kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    aggregateKernel<<<gridSize, blockSize>>>(d_input, d_output, size, agg_type, val_type);
    cudaMemcpy(&result, d_output, sizeof(AggregateResult), cudaMemcpyDeviceToHost);    
    cudaFree(d_input);
    cudaFree(d_output);
    // May be no need for this as global count could be divied on CPU
    if (agg_type == AGG_AVG && result.count > 0) {
        if (val_type == TYPE_FLOAT) {
            result.float_val /= result.count;
        } else {
            // Integer division or convert to float if needed
            if (result.int_val % result.count == 0) {
                result.int_val /= result.count;
            } else {
                result.float_val = (float)result.int_val / result.count;
            }
        }
    }
}