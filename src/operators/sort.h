#ifndef SORT_H__
#define SORT_H__

#include "../headers/device_struct.h"

void radix_sort(DeviceStruct *const d_out,
                DeviceStruct *const d_in,
                uint64_t d_in_len, unsigned int colIdx, unsigned int nCols);

#endif