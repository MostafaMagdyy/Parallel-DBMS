#ifndef JOIN_H__
#define JOIN_H__

#include "../headers/device_struct.h"
#include "../operators/join.h"

int nested_loop_join(DeviceStruct *d_table1, DeviceStruct *dtable2,
                      JoinCondition *d_conditions, int nrows1, int nrows2, int nCols1, int nCols2,
                      int nConditions, DeviceStruct *result, int shared_memory_size, int *d_offsets, int n_cols_output);

#endif
