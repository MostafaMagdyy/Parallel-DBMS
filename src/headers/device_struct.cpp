#include "device_struct.h"
#include <iostream>
#include <stdexcept>

// Constructor definition
DeviceStruct::DeviceStruct(ColumnType type, void* host_ptr, int numRows) : type(type), numRows(numRows) {
    rowSize = sizeFromColumnType(type);
    cudaError_t err;

    err = cudaMalloc(&device_ptr, numRows * rowSize);
    if (err != cudaSuccess) {
        throw "cudaMalloc failed: " + std::string(cudaGetErrorString(err));
    }

    err = cudaMemcpy(device_ptr, host_ptr, numRows * rowSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_ptr);
        throw "cudaMemcpy failed: " + std::string(cudaGetErrorString(err));
    }
}

//move constructor definition
DeviceStruct::DeviceStruct(DeviceStruct&& other) noexcept : type(other.type), device_ptr(other.device_ptr), numRows(other.numRows), rowSize(other.rowSize) {
    other.device_ptr = nullptr; // Prevent the destructor from freeing the memory
}

// Destructor definition
DeviceStruct::~DeviceStruct() {
    cudaFree(device_ptr);
}