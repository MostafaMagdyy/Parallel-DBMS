#include "../operators/sort.h"
#include "utils.h"
#include "../headers/device_struct.h"
#include "scan.h"
#include "../headers/constants.h"
#define MAX_BLOCK_SZ 128

__device__ uint64_t hostInt64ToOrderedUInt64(int64_t i)
{
    // Flip the MSB of the 64-bit signed integer
    return static_cast<uint64_t>(i) ^ 0x8000000000000000ULL;
}

__device__ uint64_t hostFloatToOrderedInt(float f)
{
    uint64_t x = 0;
    memcpy(&x, &f, sizeof(float)); // only 4 bytes are valid
    return (x & 0x80000000ULL) ? ~x : x ^ 0x80000000ULL;
}

__global__ void gpu_radix_sort_local(DeviceStruct *d_out_sorted,
                                     uint64_t *d_prefix_sums,
                                     uint64_t *d_block_sums,
                                     unsigned int input_shift_width,
                                     DeviceStruct *d_in,
                                     unsigned int d_in_len,
                                     unsigned int max_elems_per_block, unsigned int colIdx, unsigned int nCols)
{

    extern __shared__ uint64_t shmem[];
    unsigned int thid = threadIdx.x;
    uint64_t cpy_idx = max_elems_per_block * blockIdx.x + thid;

    // extern __shared__ uint64_t shmem[];
    uint64_t *s_data = shmem;
    uint64_t s_mask_out_len = max_elems_per_block + 1;
    uint64_t *s_mask_out = &s_data[max_elems_per_block];
    uint64_t *s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
    uint64_t *s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    uint64_t *s_scan_mask_out_sums = &s_mask_out_sums[4];

    if (cpy_idx < d_in_len)
    {

        ColumnType d_colType = d_in[colIdx].type;
        switch (d_colType)
        {
        case ColumnType::DATE:
            s_data[thid] = hostInt64ToOrderedUInt64(((int64_t *)d_in[colIdx].device_ptr)[cpy_idx]);
            break;
        case ColumnType::FLOAT:
            s_data[thid] = hostFloatToOrderedInt(((float *)d_in[colIdx].device_ptr)[cpy_idx]);
            break;
        default:
            s_data[thid] = 0;
            break;
        }
    }
    else
    {
        s_data[thid] = 0;
    }

    __syncthreads();

    uint64_t t_data = s_data[thid];

    uint64_t t_2bit_extract = (t_data >> input_shift_width) & 3;

    for (uint64_t i = 0; i < 4; ++i)
    {
        // Zero out s_mask_out
        s_mask_out[thid] = 0;
        if (thid == 0)
            s_mask_out[s_mask_out_len - 1] = 0;

        __syncthreads();

        // build bit mask output
        bool val_equals_i = false;
        if (cpy_idx < d_in_len)
        {
            val_equals_i = t_2bit_extract == i;
            s_mask_out[thid] = val_equals_i;
        }
        __syncthreads();

        // Scan mask outputs (Hillis-Steele)
        int partner = 0;
        uint64_t sum = 0;
        uint64_t max_steps = (uint64_t)log2f(max_elems_per_block);
        for (uint64_t d = 0; d < max_steps; d++)
        {
            partner = thid - (1 << d);
            if (partner >= 0)
            {
                sum = s_mask_out[thid] + s_mask_out[partner];
            }
            else
            {
                sum = s_mask_out[thid];
            }
            __syncthreads();
            s_mask_out[thid] = sum;
            __syncthreads();
        }

        // Shift elements to produce the same effect as exclusive scan
        uint64_t cpy_val = 0;
        cpy_val = s_mask_out[thid];
        __syncthreads();
        s_mask_out[thid + 1] = cpy_val;
        __syncthreads();

        if (thid == 0)
        {
            // Zero out first element to produce the same effect as exclusive scan
            s_mask_out[0] = 0;
            uint64_t total_sum = s_mask_out[s_mask_out_len - 1];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();

        if (val_equals_i && (cpy_idx < d_in_len))
        {
            s_merged_scan_mask_out[thid] = s_mask_out[thid];
        }

        __syncthreads();
    }

    // Scan mask output sums
    // Just do a naive scan since the array is really small
    if (thid == 0)
    {
        uint64_t run_sum = 0;
        for (uint64_t i = 0; i < 4; ++i)
        {
            s_scan_mask_out_sums[i] = run_sum;
            run_sum += s_mask_out_sums[i];
        }
    }

    __syncthreads();

    if (cpy_idx < d_in_len)
    {
        // Calculate the new indices of the input elements for sorting
        uint64_t t_prefix_sum = s_merged_scan_mask_out[thid];
        uint64_t new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];

        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        // Do this step for greater global memory transfer coalescing
        //  in next step
        s_data[new_pos] = t_data;
        for (size_t i = 0; i < nCols - 1; i++)
        {

            ColumnType d_colType = d_in[i].type;
            unsigned int g_new_pos = new_pos + (blockIdx.x * max_elems_per_block);
            switch (d_colType)
            {
            case ColumnType::DATE:
                *((int64_t *)d_out_sorted[i].device_ptr + g_new_pos) = *((int64_t *)d_in[i].device_ptr + cpy_idx);
                break;
            case ColumnType::FLOAT:
                *((float *)d_out_sorted[i].device_ptr + g_new_pos) = *((float *)d_in[i].device_ptr + cpy_idx);
                break;
            case ColumnType::STRING:
                memcpy(((char *)d_out_sorted[i].device_ptr) + g_new_pos * MAX_STRING_LENGTH, ((char *)d_in[i].device_ptr) + cpy_idx * MAX_STRING_LENGTH, MAX_STRING_LENGTH);
                break;
            default:
                break;
            }
        }

        s_merged_scan_mask_out[new_pos] = t_prefix_sum;

        __syncthreads();

        // Copy block - wise prefix sum results to global memory
        // Copy block-wise sort results to global
        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];

        *((uint64_t *)d_out_sorted[nCols - 1].device_ptr + cpy_idx) = s_data[thid];
    }
}
__global__ void gpu_glbl_shuffle(DeviceStruct *d_out,
                                 DeviceStruct *d_in,
                                 uint64_t *d_scan_block_sums,
                                 uint64_t *d_prefix_sums,
                                 uint64_t input_shift_width,
                                 unsigned int d_in_len,
                                 uint64_t max_elems_per_block, unsigned int nCols)
{
    // get d = digit
    // get n = blockIdx
    // get m = local prefix sum array value
    // calculate global position = P_d[n] + m
    // copy input element to final position in d_out

    uint64_t thid = threadIdx.x;
    uint64_t cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len)
    {
        uint64_t t_data = ((uint64_t *)d_in[nCols - 1].device_ptr)[cpy_idx];
        uint64_t t_2bit_extract = (t_data >> input_shift_width) & 3;
        uint64_t t_prefix_sum = d_prefix_sums[cpy_idx];
        uint64_t data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x] + t_prefix_sum;
        __syncthreads();
        // d_out[data_glbl_pos] = t_data;
        for (size_t i = 0; i < nCols - 1; i++)
        {

            ColumnType d_colType = d_in[i].type;
            switch (d_colType)
            {
            case ColumnType::DATE:
                *((int64_t *)d_out[i].device_ptr + data_glbl_pos) = *((int64_t *)d_in[i].device_ptr + cpy_idx);
                break;
            case ColumnType::FLOAT:
                *((float *)d_out[i].device_ptr + data_glbl_pos) = *((float *)d_in[i].device_ptr + cpy_idx);
                break;
            case ColumnType::STRING:
                memcpy(((char *)d_out[i].device_ptr) + data_glbl_pos * MAX_STRING_LENGTH, ((char *)d_in[i].device_ptr) + cpy_idx * MAX_STRING_LENGTH, MAX_STRING_LENGTH);
                break;
            default:
                break;
            }
        }
    }
}

// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
void radix_sort(DeviceStruct *const d_out,
                DeviceStruct *const d_in,
                uint64_t d_in_len, unsigned int colIdx, unsigned int nCols)
{

    std::cout << "d_in_len: " << d_in_len << std::endl;
    std::cout << "colIdx: " << colIdx << std::endl;
    std::cout << "nCols: " << nCols << std::endl;
    uint64_t block_sz = MAX_BLOCK_SZ;
    uint64_t max_elems_per_block = block_sz;
    uint64_t grid_sz = d_in_len / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (d_in_len % max_elems_per_block != 0)
        grid_sz += 1;

    uint64_t *d_prefix_sums;
    uint64_t d_prefix_sums_len = d_in_len;
    checkCudaErrors(cudaMalloc(&d_prefix_sums, sizeof(uint64_t) * d_prefix_sums_len));
    checkCudaErrors(cudaMemset(d_prefix_sums, 0, sizeof(uint64_t) * d_prefix_sums_len));

    uint64_t *d_block_sums;
    uint64_t d_block_sums_len = 4 * grid_sz; // 4-way split
    checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(uint64_t) * d_block_sums_len));
    checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(uint64_t) * d_block_sums_len));

    uint64_t *d_scan_block_sums;
    checkCudaErrors(cudaMalloc(&d_scan_block_sums, sizeof(uint64_t) * d_block_sums_len));
    checkCudaErrors(cudaMemset(d_scan_block_sums, 0, sizeof(uint64_t) * d_block_sums_len));

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    uint64_t s_data_len = max_elems_per_block;
    uint64_t s_mask_out_len = max_elems_per_block + 1;
    uint64_t s_merged_scan_mask_out_len = max_elems_per_block;
    uint64_t s_mask_out_sums_len = 4; // 4-way split
    uint64_t s_scan_mask_out_sums_len = 4;
    uint64_t shmem_sz = (s_data_len + s_mask_out_len + s_merged_scan_mask_out_len + s_mask_out_sums_len + s_scan_mask_out_sums_len) * sizeof(uint64_t);

    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (uint64_t shift_width = 0; shift_width <= 62; shift_width += 2)
    {
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_out,
                                                              d_prefix_sums,
                                                              d_block_sums,
                                                              shift_width,
                                                              d_in,
                                                              d_in_len,
                                                              max_elems_per_block, colIdx, nCols);

        // scan global block sum array
        sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

        // scatter/shuffle block-wise sorted array to final positions
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in,
                                                d_out,
                                                d_scan_block_sums,
                                                d_prefix_sums,
                                                shift_width,
                                                d_in_len,
                                                max_elems_per_block, nCols);
    }
    // checkCudaErrors(cudaMemcpy(d_out, d_in, sizeof(uint64_t) * d_in_len, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaFree(d_scan_block_sums));
    checkCudaErrors(cudaFree(d_block_sums));
    checkCudaErrors(cudaFree(d_prefix_sums));
}
