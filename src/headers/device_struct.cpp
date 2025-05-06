#include "device_struct.h"
#include <iostream>
#include <stdexcept>

// Constructor definition
DeviceStruct::DeviceStruct(ColumnType type, void* device_ptr, size_t numRows, size_t rowSize){
    this->type = type;
    this->device_ptr = device_ptr;
    this->numRows = numRows;
    this->rowSize = rowSize;
}



DeviceStruct* DeviceStruct::createStruct(ColumnType type, void* host_ptr, size_t numRows){
    size_t rowSize = sizeFromColumnType(type);
    cudaError_t err;

    void* device_ptr;

    err = cudaMalloc(&device_ptr, numRows * rowSize);
    if (err != cudaSuccess) {
        throw "cudaMalloc failed: " + std::string(cudaGetErrorString(err));
    }

    err = cudaMemcpy(device_ptr, host_ptr, numRows * rowSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_ptr);
        throw "cudaMemcpy failed: " + std::string(cudaGetErrorString(err));
    }

    return new DeviceStruct(type, device_ptr, numRows, rowSize);
}

void DeviceStruct::deleteStruct(DeviceStruct &deviceStruct){
    cudaFree(deviceStruct.device_ptr);
    deviceStruct.device_ptr = nullptr;
}

// Destructor definition
DeviceStruct::~DeviceStruct() {
    std::cout << "destroying batch";
}