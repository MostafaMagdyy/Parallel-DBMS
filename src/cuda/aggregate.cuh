#ifndef AGGREGATE_CUH
#define AGGREGATE_CUH
#include "../headers/device_struct.h"
#include "../headers/enums.h"
#include "../operators/operator_enums.h"

void tobecalledfromCPU(DeviceStruct *devicestructarr, AggregateFunctionType *arr, int opnums, void **result);
#endif // AGGREGATE_CUH