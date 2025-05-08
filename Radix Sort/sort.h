#ifndef SORT_H__
#define SORT_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.h"
#include "device_struct.h"
#include <cmath>

void radix_sort(DeviceStruct *const d_out,
                DeviceStruct *const d_in,
                uint64_t d_in_len, unsigned int colIdx, unsigned int nCols);

#endif