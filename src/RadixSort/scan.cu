#include "scan.h"

#define MAX_BLOCK_SZ 128
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

// #define ZERO_BANK_CONFLICTS

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__global__ void gpu_add_block_sums(uint64_t *const d_out,
                                   const uint64_t *const d_in,
                                   uint64_t *const d_block_sums,
                                   const size_t numElems)
{
    // uint64_t glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t d_block_sum_val = d_block_sums[blockIdx.x];

    // uint64_t d_in_val_0 = 0;
    // uint64_t d_in_val_1 = 0;

    // Simple implementation's performance is not significantly (if at all)
    //  better than previous verbose implementation
    uint64_t cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (cpy_idx < numElems)
    {
        d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
        if (cpy_idx + blockDim.x < numElems)
            d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
    }
}

// Modified version of Mark Harris' implementation of the Blelloch scan
//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
__global__ void gpu_prescan(uint64_t *const d_out,
                            const uint64_t *const d_in,
                            uint64_t *const d_block_sums,
                            const uint64_t len,
                            const uint64_t shmem_sz,
                            const uint64_t max_elems_per_block)
{
    // Allocated on invocation
    extern __shared__ uint64_t s_out[];

    int thid = threadIdx.x;
    int ai = thid;
    int bi = thid + blockDim.x;

    // Zero out the shared memory
    // Helpful especially when input size is not power of two
    s_out[thid] = 0;
    s_out[thid + blockDim.x] = 0;
    // If CONFLICT_FREE_OFFSET is used, shared memory size
    //  must be a 2 * blockDim.x + blockDim.x/num_banks
    s_out[thid + blockDim.x + (blockDim.x >> LOG_NUM_BANKS)] = 0;

    __syncthreads();

    // Copy d_in to shared memory
    // Note that d_in's elements are scattered into shared memory
    //  in light of avoiding bank conflicts
    uint64_t cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
    if (cpy_idx < len)
    {
        s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
        if (cpy_idx + blockDim.x < len)
            s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
    }

    // For both upsweep and downsweep:
    // Sequential indices with conflict free padding
    //  Amount of padding = target index / num banks
    //  This "shifts" the target indices by one every multiple
    //   of the num banks
    // offset controls the stride and starting index of
    //  target elems at every iteration
    // d just controls which threads are active
    // Sweeps are pivoted on the last element of shared memory

    // Upsweep/Reduce step
    int offset = 1;
    for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_out[bi] += s_out[ai];
        }
        offset <<= 1;
    }

    // Save the total sum on the global block sums array
    // Then clear the last element on the shared memory
    if (thid == 0)
    {
        d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
        s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
    }

    // Downsweep step
    for (int d = 1; d < max_elems_per_block; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            uint64_t temp = s_out[ai];
            s_out[ai] = s_out[bi];
            s_out[bi] += temp;
        }
    }
    __syncthreads();

    // Copy contents of shared memory to global memory
    if (cpy_idx < len)
    {
        d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
        if (cpy_idx + blockDim.x < len)
            d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
    }
}

void sum_scan_blelloch(uint64_t *const d_out,
                       const uint64_t *const d_in,
                       const size_t numElems)
{
    // Zero out d_out
    checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(uint64_t)));

    // Set up number of threads and blocks

    uint64_t block_sz = MAX_BLOCK_SZ / 2;
    uint64_t max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

    // If input size is not power of two, the remainder will still need a whole block
    // Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
    // uint64_t grid_sz = (uint64_t) std::ceil((double) numElems / (double) max_elems_per_block);
    // UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically
    //  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
    uint64_t grid_sz = numElems / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (numElems % max_elems_per_block != 0)
        grid_sz += 1;

    // Conflict free padding requires that shared memory be more than 2 * block_sz
    uint64_t shmem_sz = max_elems_per_block + ((max_elems_per_block) >> LOG_NUM_BANKS);

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    uint64_t *d_block_sums;
    checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(uint64_t) * grid_sz));
    checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(uint64_t) * grid_sz));

    // Sum scan data allocated to each block
    // gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(uint64_t) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
    gpu_prescan<<<grid_sz, block_sz, sizeof(uint64_t) * shmem_sz>>>(d_out,
                                                                    d_in,
                                                                    d_block_sums,
                                                                    numElems,
                                                                    shmem_sz,
                                                                    max_elems_per_block);

    // Sum scan total sums produced by each block
    // Use basic implementation if number of total sums is <= 2 * block_sz
    //  (This requires only one block to do the scan)
    if (grid_sz <= max_elems_per_block)
    {
        uint64_t *d_dummy_blocks_sums;
        checkCudaErrors(cudaMalloc(&d_dummy_blocks_sums, sizeof(uint64_t)));
        checkCudaErrors(cudaMemset(d_dummy_blocks_sums, 0, sizeof(uint64_t)));
        // gpu_sum_scan_blelloch<<<1, block_sz, sizeof(uint64_t) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
        gpu_prescan<<<1, block_sz, sizeof(uint64_t) * shmem_sz>>>(d_block_sums,
                                                                  d_block_sums,
                                                                  d_dummy_blocks_sums,
                                                                  grid_sz,
                                                                  shmem_sz,
                                                                  max_elems_per_block);
        
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaFree(d_dummy_blocks_sums));
    }
    // Else, recurse on this same function as you'll need the full-blown scan
    //  for the block sums
    else
    {
        uint64_t *d_in_block_sums;
        checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(uint64_t) * grid_sz));
        checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(uint64_t) * grid_sz, cudaMemcpyDeviceToDevice));
        sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
        checkCudaErrors(cudaFree(d_in_block_sums));
    }
    gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

    checkCudaErrors(cudaFree(d_block_sums));
}
