#ifndef AGGREGATE_CUH
#define AGGREGATE_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Enum for aggregation types
enum AggregateType {
    AGG_MIN = 0,
    AGG_MAX = 1,
    AGG_SUM = 2,
    AGG_AVG = 3,
    AGG_COUNT = 4
};

// Enum for value types
enum ValueType {
    TYPE_INT = 0,
    TYPE_FLOAT = 1,
    TYPE_DATE = 2 
};

// Structure to hold the aggregate result
struct AggregateResult {
    union {
        float float_val;
        int64_t int_val;
    };
    int64_t count;  // For AVF (May be no need for it later)
};

// Device function declarations
__device__ float atomicMaxFloat(float* address, float val);
__device__ float atomicMinFloat(float* address, float val);
__device__ int64_t atomicAddInt64(int64_t* address, int64_t val);
__device__ int64_t atomicMaxInt64(int64_t* address, int64_t val);
__device__ int64_t atomicMinInt64(int64_t* address, int64_t val);

// Warp reduction functions
__device__ void warpReduce(float& value, int aggregation_type);
__device__ void warpReduce(int64_t& value, int aggregation_type);

// Kernel declaration
__global__ void aggregateKernel(void* input, AggregateResult* output, 
                             int size, int aggregation_type, int value_type);

// Host function declaration
void computeAggregate(void* data, int size, AggregateType agg_type, 
                     ValueType val_type, AggregateResult& result);

#endif // AGGREGATE_CUH