#include <thrust/scan.h>
#include "device_struct.h"
#include "join.h"
#include <stdio.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
__device__ unsigned int d_global_row_count = 0;

__device__ bool evaluate_single_condition(
    const void *pVal1,
    const void *pVal2,
    ColumnType type_for_comparison,
    ComparisonOperator op)
{
    switch (type_for_comparison)
    {
    case ColumnType::INT:
    {
        int64_t val1 = *static_cast<const int64_t *>(pVal1);
        int64_t val2 = *static_cast<const int64_t *>(pVal2);
        // if (threadIdx.x == 0)
        // {
        //     printf("val1: %lld, val2: %ld\n", val1, val2);
        // }
        switch (op)
        {
        case ComparisonOperator::EQUALS:
            return val1 == val2;
        case ComparisonOperator::NOT_EQUALS:
            return val1 != val2;
        case ComparisonOperator::LESS_THAN:
            return val1 < val2;
        case ComparisonOperator::LESS_THAN_EQUALS:
            return val1 <= val2;
        case ComparisonOperator::GREATER_THAN:
            return val1 > val2;
        case ComparisonOperator::GREATER_THAN_EQUALS:
            return val1 >= val2;
        default:
            return false;
        }
        break; // Added break
    }
    case ColumnType::FLOAT:
    {
        float val1 = *static_cast<const float *>(pVal1);
        float val2 = *static_cast<const float *>(pVal2);
        switch (op)
        {
        case ComparisonOperator::EQUALS:
            return val1 == val2;
        case ComparisonOperator::NOT_EQUALS:
            return val1 != val2;
        case ComparisonOperator::LESS_THAN:
            return val1 < val2;
        case ComparisonOperator::LESS_THAN_EQUALS:
            return val1 <= val2;
        case ComparisonOperator::GREATER_THAN:
            return val1 > val2;
        case ComparisonOperator::GREATER_THAN_EQUALS:
            return val1 >= val2;
        default:
            return false;
        }
        break; // Added break
    }
    case ColumnType::UINT64:
    {
        uint64_t val1 = *static_cast<const uint64_t *>(pVal1);
        uint64_t val2 = *static_cast<const uint64_t *>(pVal2);
        switch (op)
        {
        case ComparisonOperator::EQUALS:
            return val1 == val2;
        case ComparisonOperator::NOT_EQUALS:
            return val1 != val2;
        case ComparisonOperator::LESS_THAN:
            return val1 < val2;
        case ComparisonOperator::LESS_THAN_EQUALS:
            return val1 <= val2;
        case ComparisonOperator::GREATER_THAN:
            return val1 > val2;
        case ComparisonOperator::GREATER_THAN_EQUALS:
            return val1 >= val2;
        default:
            return false;
        }
        break;
    }
    default:
        return false;
    }
    return false;
}

// Phase 1: Count matches per row in table1
__global__ void count_nested_loop_join_kernel(
    DeviceStruct *d_table1, DeviceStruct *d_table2,
    JoinCondition *d_conditions, int nrows1, int nrows2,
    int nCols1, int nCols2, int nConditions,
    int *d_counts, int *d_offsets)
{
    extern __shared__ uint8_t shmem[];
    int thid = threadIdx.x;
    int gthid = blockIdx.x * blockDim.x + thid;
    if (gthid >= nrows1)
        return;

    // Load table2 data into shared memory
    for (int col = 0; col < nCols2; col++)
    {
        ColumnType type = d_table2[col].type;
        for (int j = 0; j < nrows2; j += blockDim.x)
        {
            if (thid + j >= nrows2)
                break;
            switch (type)
            {
            case ColumnType::INT:
            {
                int64_t val = ((const int64_t *)d_table2[col].device_ptr)[thid + j];
                *((int64_t *)(shmem + d_offsets[col]) + j + thid) = val;
                break;
            }
            case ColumnType::FLOAT:
            {
                float val = ((const float *)d_table2[col].device_ptr)[thid + j];
                *((float *)(shmem + d_offsets[col]) + j + thid) = val;
                break;
            }
            default:
                break;
            }
        }
    }
    __syncthreads();

    int match_count = 0;
    for (int i = 0; i < nrows2; i++)
    {
        bool match = true;
        for (int c = 0; c < nConditions; c++)
        {
            const auto &cond = d_conditions[c];
            const void *left_val = (char *)d_table1[cond.leftColumnIdx].device_ptr +
                                   gthid * d_table1[cond.leftColumnIdx].element_size;
            const void *right_val = shmem + d_offsets[cond.rightColumnIdx] +
                                    i * d_table2[cond.rightColumnIdx].element_size;

            if (!evaluate_single_condition(left_val, right_val,
                                           cond.columnType, cond.op))
            {
                match = false;
                break;
            }
        }
        if (match)
            match_count++;
    }

    d_counts[gthid] = match_count;
}

// Phase 2: Write results using precomputed indices
__global__ void write_nested_loop_join_kernel(
    DeviceStruct *d_table1, DeviceStruct *d_table2,
    JoinCondition *d_conditions, int nrows1, int nrows2,
    int nCols1, int nCols2, int nConditions,
    DeviceStruct *result, int *d_offsets, int n_cols_output,
    int *d_prefix_sum)
{
    extern __shared__ uint8_t shmem[];
    int thid = threadIdx.x;
    int gthid = blockIdx.x * blockDim.x + thid;
    if (gthid >= nrows1)
        return;

    // Load table2 data into shared memory (same as count phase)
    for (int col = 0; col < nCols2; col++)
    {
        ColumnType type = d_table2[col].type;
        for (int j = 0; j < nrows2; j += blockDim.x)
        {
            if (thid + j >= nrows2)
                break;
            switch (type)
            {
            case ColumnType::INT:
            {
                int64_t val = ((const int64_t *)d_table2[col].device_ptr)[thid + j];
                *((int64_t *)(shmem + d_offsets[col]) + j + thid) = val;
                break;
            }
            case ColumnType::FLOAT:
            {
                float val = ((const float *)d_table2[col].device_ptr)[thid + j];
                *((float *)(shmem + d_offsets[col]) + j + thid) = val;
                break;
            }
            default:
                break;
            }
        }
    }
    __syncthreads();

    int write_pos = d_prefix_sum[gthid];
    int matches_written = 0;

    for (int i = 0; i < nrows2; i++)
    {
        bool match = true;
        for (int c = 0; c < nConditions; c++)
        {
            const auto &cond = d_conditions[c];
            const void *left_val = (char *)d_table1[cond.leftColumnIdx].device_ptr +
                                   gthid * d_table1[cond.leftColumnIdx].element_size;
            const void *right_val = shmem + d_offsets[cond.rightColumnIdx] +
                                    i * d_table2[cond.rightColumnIdx].element_size;

            if (!evaluate_single_condition(left_val, right_val,
                                           cond.columnType, cond.op))
            {
                match = false;
                break;
            }
        }

        if (match)
        {
            // Modified join_row without atomic
            int curr_r = write_pos + matches_written++;
            int currCol = 0;

            // Copy table1 columns
            for (int c1 = 0; c1 < nCols1; c1++)
            {
                ColumnType type = d_table1[c1].type;
                void *src = (char *)d_table1[c1].device_ptr +
                            gthid * d_table1[c1].element_size;
                void *dest = (char *)result[currCol].device_ptr +
                             curr_r * result[currCol].element_size;
                memcpy(dest, src, d_table1[c1].element_size);
                currCol++;
            }

            // Copy table2 columns
            for (int c2 = 0; c2 < nCols2; c2++)
            {
                bool skip = false;
                for (int c = 0; c < nConditions; c++)
                {
                    if (d_conditions[c].rightColumnIdx == c2 &&
                        d_conditions[c].op == ComparisonOperator::EQUALS)
                    {
                        skip = true;
                        break;
                    }
                }
                if (skip)
                    continue;

                ColumnType type = d_table2[c2].type;
                void *src = shmem + d_offsets[c2] + i * d_table2[c2].element_size;
                void *dest = (char *)result[currCol].device_ptr +
                             curr_r * result[currCol].element_size;
                memcpy(dest, src, d_table2[c2].element_size);
                currCol++;
            }
        }
    }
}

void nested_loop_join(DeviceStruct *d_table1, DeviceStruct *d_table2,
                      JoinCondition *d_conditions, int nrows1, int nrows2,
                      int nCols1, int nCols2, int nConditions,
                      DeviceStruct *result, int shared_memory_size,
                      int *d_offsets, int n_cols_output)
{
    int *d_counts, *d_prefix_sum;
    cudaMalloc(&d_counts, nrows1 * sizeof(int));
    cudaMemset(d_counts, 0, nrows1 * sizeof(int));

    // Phase 1: Count matches
    int blocks = (nrows1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_nested_loop_join_kernel<<<blocks, BLOCK_SIZE, shared_memory_size>>>(
        d_table1, d_table2, d_conditions, nrows1, nrows2,
        nCols1, nCols2, nConditions, d_counts, d_offsets);

    // Phase 2: Compute prefix sum
    cudaMalloc(&d_prefix_sum, (nrows1 + 1) * sizeof(int));
    thrust::exclusive_scan(thrust::device, d_counts, d_counts + nrows1,
                           d_prefix_sum, 0);

    // Phase 3: Write results
    write_nested_loop_join_kernel<<<blocks, BLOCK_SIZE, shared_memory_size>>>(
        d_table1, d_table2, d_conditions, nrows1, nrows2,
        nCols1, nCols2, nConditions, result, d_offsets,
        n_cols_output, d_prefix_sum);

    cudaFree(d_counts);
    cudaFree(d_prefix_sum);
}