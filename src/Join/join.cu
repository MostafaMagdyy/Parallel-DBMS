#include "../headers/device_struct.h"
#include "join.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "../headers/column.h"
#include "../headers/constants.h"
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
    case ColumnType::DATE:
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
        // printf("val1: %f, val2: %f\n", val1, val2);
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
        case ColumnType::DATE:
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
        case ColumnType::STRING:
        {
            char *str_start = ((char *)d_table1[col1].device_ptr) + gthid * MAX_STRING_LENGTH;
            memcpy(((char *)result[currCol].device_ptr) + curr_r * MAX_STRING_LENGTH, str_start, MAX_STRING_LENGTH);
            break;
        }
        default:
            continue; // Skip if type_for_this_condition is not handled
        }
        currCol++;
    }
    // 2. Copy all columns from table2's inner_row_in_tile_idx (from shared memory)
    for (int c2 = 0; c2 < n_cols2; ++c2)
    {
        ColumnType type = d_table2[c2].type;
        switch (type)
        {
        case ColumnType::DATE:
            // *(static_cast<int64_t *>(
            //     static_cast<void *>(&result[currCol].device_ptr[curr_r]))) =
            //     *(static_cast<const int64_t *>(
            //           static_cast<const void *>(shmem)) +
            //       i + (offset[col2] / sizeof(int64_t)));
            ((int64_t *)result[currCol].device_ptr)[curr_r] = *((const int64_t *)((shmem)) + i + (d_offsets[c2] / sizeof(int64_t))); // this is incorrect because if the previous offset is not divisible by 8, then the result will be incorrect
            break;

        case ColumnType::FLOAT:
            // *(static_cast<float *>(
            //     static_cast<void *>(&result[currCol].device_ptr[curr_r]))) =
            //     *(static_cast<const float *>(
            //           static_cast<const void *>(shmem)) +
            //       i + (offset[col2] / sizeof(float)));
            ((float *)result[currCol].device_ptr)[curr_r] = *((const float *)((shmem)) + i + (d_offsets[c2] / sizeof(float))); // this is incorrect because if the previous offset is not divisible by 4, then the result will be incorrect

            break;
        case ColumnType::STRING:
        {
            char *str_start = ((char *)shmem) + i * MAX_STRING_LENGTH + (d_offsets[c2] / sizeof(char));
            // printf("copying string from %d to %d\n", i, curr_r);
            // printf("string: %s\n", str_start);
            memcpy(((char *)result[currCol].device_ptr) + curr_r * MAX_STRING_LENGTH, str_start, MAX_STRING_LENGTH);
            break;
        }

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
    if (thid < 50 && blockIdx.x == 0)
    {
        // printf("priting from thread %d\n", thid);
    }

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

            case ColumnType::DATE:
                // printf("indx %d\n", (int)(thid + j + d_offsets[i] / sizeof(float)));
                // printf("offset %d\n", (int)(d_offsets[i] / sizeof(float)));
                // printf("thid %d\n", (int)thid);
                // printf("j %d\n", (int)j);
                // printf("i %d\n", (int)i);
                // printf("value %f\n", ((const float *)d_table2[i].device_ptr)[thid + j]);

                // LHS: Treat shmem as base, add element offset (thid + j + byte_offset_to_col_start / element_size)
                // RHS: Treat d_table2[i].device_ptr as base of int64_t array, get (thid + j)-th element
                *((int64_t *)(shmem) + thid + j + d_offsets[i] / sizeof(int64_t)) =
                    ((const int64_t *)d_table2[i].device_ptr)[thid + j];
                break;

            case ColumnType::FLOAT:
                // LHS: Treat shmem as base, add element offset (thid + j + byte_offset_to_col_start / element_size)
                // RHS: Treat d_table2[i].device_ptr as base of float array, get (thid + j)-th element
                // printf("indx %d\n", (int)(thid + j + d_offsets[i] / sizeof(float)));
                // printf("offset %d\n", (int)(d_offsets[i] / sizeof(float)));
                // printf("thid %d\n", (int)thid);
                // printf("j %d\n", (int)j);
                // printf("i %d\n", (int)i);
                // printf("value %f\n", ((const float *)d_table2[i].device_ptr)[thid + j]);
                *((float *)(shmem) + thid + j + d_offsets[i] / sizeof(float)) =
                    ((const float *)d_table2[i].device_ptr)[thid + j];
                break;
            case ColumnType::STRING:
            {
                char *str_start = ((char *)d_table2[i].device_ptr) + (thid + j) * MAX_STRING_LENGTH;
                memcpy(((char *)shmem) + (thid + j) * MAX_STRING_LENGTH + d_offsets[i] / sizeof(char), str_start, MAX_STRING_LENGTH);
                break;
            }
            default:
                break;
            }
        }
    }
    if (gthid >= nrows1)
    {
        return;
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
    //             case ColumnType::DATE:
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
            case ColumnType::DATE:
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

                // printf("nrows 2 %d\n", (int)i);
                // printf("Addres %lld\n", (long long)(static_cast<const float *>(
                //                                         static_cast<const void *>(shmem)) +
                //                                     i +
                //                                     (d_offsets[d_conditions[z].rightColumnIdx] / sizeof(float))));
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

int nested_loop_join(DeviceStruct *d_table1, DeviceStruct *d_table2,
                     JoinCondition *d_conditions, int nrows1, int nrows2, int nCols1, int nCols2,
                     int nConditions, DeviceStruct *result, int shared_memory_size, int *d_offsets, int n_cols_output)
{
    // Reset the counter
    unsigned int zero = 0;
    cudaMemcpyToSymbol(d_global_row_count, &zero, sizeof(unsigned int));

    int nblocks = (nrows1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // static int launch_count = 0;
    // printf("Launch #%d\n", ++launch_count);
    // std::cout << "calling Kernel" << std::endl;
    // printf("Block size: %d\n", BLOCK_SIZE);
    // printf("Grid size: %d\n", nblocks);
    // printf("Shared memory: %d\n", shared_memory_size);
    // printf("d_table1: %p\n", (void *)d_table1);
    // printf("d_table2: %p\n", (void *)d_table2);
    // printf("d_conditions: %p\n", (void *)d_conditions);
    // printf("result: %p\n", (void *)result);
    // printf("d_offsets: %p\n", (void *)d_offsets);
    nested_loop_join_kernel<<<nblocks, BLOCK_SIZE, shared_memory_size>>>(
        d_table1, d_table2, d_conditions, nrows1, nrows2, nCols1, nCols2,
        nConditions, result, d_offsets, n_cols_output);
    // cudaDeviceSynchronize();
    // printf("kernel 1 finished\n");
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        printf("Kernel launch failed with error \"%s\"\n", cudaGetErrorString(cudaError));
        printf("shared memory size: %d\n", shared_memory_size);
        printf("nrows1: %d\n", nrows1);
        printf("nrows2: %d\n", nrows2);
        printf("nCols1: %d\n", nCols1);
        printf("nCols2: %d\n", nCols2);
        printf("nConditions: %d\n", nConditions);
        printf("n_cols_output: %d\n", n_cols_output);
    }

    // Get the result
    unsigned int actual_rows_out;
    cudaMemcpyFromSymbol(&actual_rows_out, d_global_row_count, sizeof(unsigned int));
    return actual_rows_out;
}
