#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device = 0, val = 0;
    cudaSetDevice(device);

    cudaDeviceGetAttribute(&val,
        cudaDevAttrMaxSharedMemoryPerBlock,
        device);
    std::cout << "Max shared memory per block: "
              << (val/1024) << " KB\n";

    cudaDeviceGetAttribute(&val,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor,
        device);
    std::cout << "Max shared memory per SM:    "
              << (val/1024) << " KB\n";
    return 0;
}
