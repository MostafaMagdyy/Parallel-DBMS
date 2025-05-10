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
__device__ void join_row(
    DeviceStruct *d_table1, DeviceStruct *d_table2, uint8_t *shmem, int n_cols_output, int n_cols1, int n_cols2,
    int n_rows1, int n_rows2, int nConditions, int *d_offsets, DeviceStruct *result,
    JoinCondition *d_conditions, int i, int gthid)
{
    int curr_r = atomicAdd(&d_global_row_count, 1);
    // int temp = atomicAdd(&d_global_row_count, 0);
    // if (threadIdx.x == 0)
    // {
    //     printf("curr_r: %d\n", curr_r);
    // }
    int currCol = 0;
    for (size_t col1 = 0; col1 < n_cols1; col1++)
    {
        ColumnType type = d_table1[col1].type;
        switch (type)
        {
        case ColumnType::INT:
            // *(static_cast<int64_t *>(
            //     static_cast<void *>(&result[currCol].device_ptr[curr_r]))) =
            //     *(static_cast<const int64_t *>(
            //         static_cast<const void *>(&d_table1[col1].device_ptr[gthid])));
            ((int64_t *)result[currCol].device_ptr)[curr_r] = ((const int64_t *)d_table1[col1].device_ptr)[gthid];
            break;

        case ColumnType::FLOAT:
            // *(static_cast<float *>(
            //     static_cast<void *>(&result[currCol].device_ptr[curr_r]))) =
            //     *(static_cast<const float *>(
            //         static_cast<const void *>(&d_table1[col1].device_ptr[gthid])));
            ((float *)result[currCol].device_ptr)[curr_r] = ((const float *)d_table1[col1].device_ptr)[gthid];
            break;

        default:
            continue; // Skip if type_for_this_condition is not handled
        }
        currCol++;
    }
    // 2. Copy all columns from table2's inner_row_in_tile_idx (from shared memory)
    for (int c2 = 0; c2 < n_cols2; ++c2)
    {
        bool skip = false;
        for (int z = 0; z < nConditions; ++z)
        {
            if (d_conditions[z].op != ComparisonOperator::EQUALS)
            {
                continue;
            }
            if (d_conditions[z].rightColumnIdx == c2)
            {

                skip = true;
                break;
            }
        }
        if (skip)
            continue;
        ColumnType type = d_table2[c2].type;
        switch (type)
        {
        case ColumnType::INT:
            // *(static_cast<int64_t *>(
            //     static_cast<void *>(&result[currCol].device_ptr[curr_r]))) =
            //     *(static_cast<const int64_t *>(
            //           static_cast<const void *>(shmem)) +
            //       i + (offset[col2] / sizeof(int64_t)));
            ((int64_t *)result[currCol].device_ptr)[curr_r] = *((const int64_t *)((shmem)) + i + (d_offsets[c2] / sizeof(int64_t)));
            break;

        case ColumnType::FLOAT:
            // *(static_cast<float *>(
            //     static_cast<void *>(&result[currCol].device_ptr[curr_r]))) =
            //     *(static_cast<const float *>(
            //           static_cast<const void *>(shmem)) +
            //       i + (offset[col2] / sizeof(float)));
            ((float *)result[currCol].device_ptr)[curr_r] = *((const float *)((shmem)) + i + (d_offsets[c2] / sizeof(float)));

            break;

        default:
            continue; // Skip if type_for_this_condition is not handled for shared memory access
        }

        currCol++;
    }

} // namespace join_row

__global__ void nested_loop_join_kernel(DeviceStruct *d_table1, DeviceStruct *d_table2,
                                        JoinCondition *d_conditions, int nrows1, int nrows2, int nCols1, int nCols2,
                                        int nConditions, DeviceStruct *result, int *d_offsets, int n_cols_output)
{

    extern __shared__ uint8_t shmem[];
    int thid = threadIdx.x;
    int gthid = blockIdx.x * blockDim.x + thid;
    // if (thid == 0 && blockIdx.x == 0)
    // {
    //     for (int i = 0; i < nrows2; i++)
    //     {
    //         for (int j = 0; j < nCols2; j++)
    //         {
    //             ColumnType type = d_table2[j].type;
    //             switch (type)
    //             {
    //             case ColumnType::INT:
    //                 printf("d_table2[%d].device_ptr: %lld, ", j, ((const int64_t *)d_table2[j].device_ptr)[i]);
    //                 break;
    //             case ColumnType::FLOAT:
    //                 printf("d_table2[%d].device_ptr: %f, ", j, ((const float *)d_table2[j].device_ptr)[i]);
    //                 break;
    //             default:
    //                 break;
    //             }
    //         }
    //         printf("\n");
    //     }
    // }
    if (gthid >= nrows1)
    {
        return;
    }
    for (size_t i = 0; i < nCols2; i++)
    {
        for (size_t j = 0; j < nrows2; j += blockDim.x)
        {
            if (thid + j >= nrows2)
                break;
            ColumnType type = d_table2[i].type;
            switch (type)
            {
                // Assuming d_table2[i].device_ptr is something like char* or void*
                // and shmem is also something like char* or void*

            case ColumnType::INT:
                // LHS: Treat shmem as base, add element offset (thid + j + byte_offset_to_col_start / element_size)
                // RHS: Treat d_table2[i].device_ptr as base of int64_t array, get (thid + j)-th element
                *((int64_t *)(shmem) + thid + j + d_offsets[i] / sizeof(int64_t)) =
                    ((const int64_t *)d_table2[i].device_ptr)[thid + j];
                break;

            case ColumnType::FLOAT:
                // LHS: Treat shmem as base, add element offset (thid + j + byte_offset_to_col_start / element_size)
                // RHS: Treat d_table2[i].device_ptr as base of float array, get (thid + j)-th element
                *((float *)(shmem) + thid + j + d_offsets[i] / sizeof(float)) =
                    ((const float *)d_table2[i].device_ptr)[thid + j];
                break;
            default:
                break;
            }
        }
    }
    __syncthreads();
    // if (thid == 0 && blockIdx.x == 0)
    // {
    //     for (int i = 0; i < nrows2; i++)
    //     {
    //         for (int j = 0; j < nCols2; j++)
    //         {
    //             ColumnType type = d_table2[j].type;
    //             switch (type)
    //             {
    //             case ColumnType::INT:
    //                 printf("d_table2[%d].device_ptr: %lld, ", j, *((int64_t *)(shmem) + i + d_offsets[j] / sizeof(int64_t)));
    //                 break;
    //             case ColumnType::FLOAT:
    //                 printf("d_table2[%d].device_ptr: %f, ", j, *((float *)(shmem) + i + d_offsets[j] / sizeof(float)));
    //                 break;
    //             default:
    //                 break;
    //             }
    //         }
    //         printf("\n");
    //     }
    // }
    for (size_t i = 0; i < nrows2; i++)
    {

        bool is_match = true;
        for (int z = 0; z < nConditions; z++)
        {

            ColumnType type = d_conditions[z].columnType;
            ComparisonOperator op = d_conditions[z].op;
            switch (type)
            {
            case ColumnType::INT:
                // if (thid == 0)
                // {
                //     printf("Condition %d: %lld\n", z, *(static_cast<const int64_t *>(static_cast<const void *>(shmem)) + i + (d_offsets[d_conditions[z].rightColumnIdx] / sizeof(int64_t))));
                // }
                is_match = is_match &&
                           evaluate_single_condition(
                               ((const int64_t *)d_table1[d_conditions[z].leftColumnIdx].device_ptr) + gthid,
                               (static_cast<const int64_t *>(
                                    static_cast<const void *>(shmem)) +
                                i +
                                (d_offsets[d_conditions[z].rightColumnIdx] / sizeof(int64_t))),
                               type,
                               op);
                break;

            case ColumnType::FLOAT:
                is_match = is_match &&
                           evaluate_single_condition(
                               // Corrected: Cast device_ptr to const float* then add element offset gthid
                               ((const float *)d_table1[d_conditions[z].leftColumnIdx].device_ptr) + gthid,
                               // This part was already okay:
                               (static_cast<const float *>(
                                    static_cast<const void *>(shmem)) + // Base pointer from shmem
                                i +
                                (d_offsets[d_conditions[z].rightColumnIdx] / sizeof(float))), // Element offset in shmem
                               type,
                               op);
                break;
            default:
                break;
            }
            if (!is_match)
            {
                // if (thid == 0)
                //     printf("Condition not met for row %d\n", i);
                // break;
            }
        }
        if (is_match)
        {
            join_row(d_table1, d_table2, shmem, n_cols_output, nCols1, nCols2,
                     nrows1, nrows2, nConditions, d_offsets, result, d_conditions, i, gthid);
        }
    }
}

void nested_loop_join(DeviceStruct *d_table1, DeviceStruct *d_table2,
                      JoinCondition *d_conditions, int nrows1, int nrows2, int nCols1, int nCols2,
                      int nConditions, DeviceStruct *result, int shared_memory_size, int *d_offsets, int n_cols_output)

{

    int nblocks = (nrows1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    nested_loop_join_kernel<<<nblocks, BLOCK_SIZE, shared_memory_size>>>(
        d_table1, d_table2, d_conditions, nrows1, nrows2, nCols1, nCols2,
        nConditions, result, d_offsets, n_cols_output);
}
