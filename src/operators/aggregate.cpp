#include "aggregate.h"
#include <iostream>
#include <utility>
#include <string>
#include <cuda_runtime.h>
#include "../headers/device_struct.h"
#include "../cuda/aggregate.cuh"
#include <cfloat>

AggregateFunctionType parseAggregateExpression(const std::string &name)
{
    size_t openParen = name.find('(');
    size_t closeParen = name.find(')', openParen + 1);

    if (openParen == std::string::npos || closeParen == std::string::npos)
    {
        throw "Invalid aggregate expression format";
    }

    std::string functionName = name.substr(0, openParen);

    AggregateFunctionType function;
    if (functionName == "sum")
    {
        function = AggregateFunctionType::SUM;
    }
    else if (functionName == "avg")
    {
        function = AggregateFunctionType::AVG;
    }
    else if (functionName == "count")
    {
        function = AggregateFunctionType::COUNT;
    }
    else if (functionName == "min")
    {
        function = AggregateFunctionType::MIN;
    }
    else if (functionName == "max")
    {
        function = AggregateFunctionType::MAX;
    }
    else
    {
        throw "Unknown aggregate function: " + functionName;
    }
    return function;
}

std::vector<void *> aggregate(std::shared_ptr<Table> table, std::vector<AggregateFunctionType> aggregate_functions, std::vector<std::string> column_names)
{

    // TODO add avg by using sum and count of rows by batches
    std::unordered_map<std::string, std::vector<AggregateFunctionType>> column_aggregate_functions;
    std::unordered_map<std::string, std::vector<void *>> column_aggregate_results;
    for (size_t i = 0; i < column_names.size(); i++)
    {
        if (column_aggregate_functions.find(column_names[i]) == column_aggregate_functions.end())
        {
            column_aggregate_functions[column_names[i]] = std::vector<AggregateFunctionType>();
            column_aggregate_results[column_names[i]] = std::vector<void *>();
        }
        if (std::find(column_aggregate_functions[column_names[i]].begin(), column_aggregate_functions[column_names[i]].end(), aggregate_functions[i]) == column_aggregate_functions[column_names[i]].end())
        {
            column_aggregate_functions[column_names[i]].push_back(aggregate_functions[i]);

            switch (table->getColumnType(column_names[i]))
            {
            case ColumnType::FLOAT:
                switch (aggregate_functions[i])
                {
                case AggregateFunctionType::SUM:
                    column_aggregate_results[column_names[i]].push_back(new float(0));
                    break;
                case AggregateFunctionType::MAX:
                    column_aggregate_results[column_names[i]].push_back(new float(-FLT_MAX));
                    break;
                case AggregateFunctionType::MIN:
                    column_aggregate_results[column_names[i]].push_back(new float(FLT_MAX));
                    break;
                }
                break;
            case ColumnType::DATE:
                switch (aggregate_functions[i])
                {
                case AggregateFunctionType::MAX:
                    column_aggregate_results[column_names[i]].push_back(new int64_t(INT64_MIN));
                    break;
                case AggregateFunctionType::MIN:
                    column_aggregate_results[column_names[i]].push_back(new int64_t(INT64_MAX));
                    break;
                }
                break;
            default:
                throw "Unsupported column type for aggregate function: " + column_names[i];
            }
        }
    }

    auto timeSum = 0, sum_2 = 0, copy_time = 0, kernel_time = 0;

    while (table->hasMoreData())
    {
        table->readNextBatch();
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<DeviceStruct> device_struct_ptrs = table->transferBatchToGPU();
        std::vector<std::shared_ptr<ColumnBatch>> current_batch = table->getCurrentBatch();
        for (auto &[column_name, aggregate_functions_column] : column_aggregate_functions)
        {
            std::vector<AggregateFunctionType> aggregate_functions;
            // TODO: create device struct for this column
            size_t projected_index = table->getColumnIndexProjected(column_name);
            std::shared_ptr<ColumnBatch> column_batch = current_batch[projected_index];
            auto start = std::chrono::high_resolution_clock::now();
            column_batch->transferToGPU();
            auto end = std::chrono::high_resolution_clock::now();
            copy_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            DeviceStruct device_struct_ptr = device_struct_ptrs[projected_index];
            for (auto &aggregate_function : aggregate_functions_column)
            {
                device_struct_ptrs.push_back(device_struct_ptr);
                aggregate_functions.push_back(aggregate_function);
                // use device struct for multiple aggregate `functions
                //  TODO
            }
            std::vector<void *> batch_aggregate_results(aggregate_functions.size());
            for (size_t i = 0; i < aggregate_functions.size(); i++)
            {
                switch (table->getColumnType(column_name))
                {
                case ColumnType::FLOAT:
                    batch_aggregate_results[i] = new float(0);
                    break;
                case ColumnType::DATE:
                    batch_aggregate_results[i] = new int64_t(INT64_MIN);
                    break;
                default:
                    throw "Unsupported column type for aggregate function: " + column_names[i];
                }
            }
            tobecalledfromCPU(device_struct_ptrs.data(), aggregate_functions.data(), aggregate_functions.size(), batch_aggregate_results.data());
            end = std::chrono::high_resolution_clock::now();
            kernel_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            for (size_t i = 0; i < column_aggregate_results[column_name].size(); i++)
            {
                auto aggregate_result = column_aggregate_results[column_name][i];
                switch (table->getColumnType(column_name))
                {
                case ColumnType::FLOAT:
                    switch (column_aggregate_functions[column_name][i])
                    {
                    case AggregateFunctionType::SUM:
                        *((float *)aggregate_result) += *(float *)batch_aggregate_results[i];
                        break;
                    case AggregateFunctionType::MAX:
                        *((float *)aggregate_result) = std::max(*((float *)aggregate_result), *(float *)batch_aggregate_results[i]);
                        break;
                    case AggregateFunctionType::MIN:
                        *((float *)aggregate_result) = std::min(*((float *)aggregate_result), *(float *)batch_aggregate_results[i]);
                        break;
                    default:
                        throw "Unsupported aggregate function for float column: ";
                    }
                    break;
                case ColumnType::DATE:
                    switch (column_aggregate_functions[column_name][i])
                    {
                    case AggregateFunctionType::MAX:
                        *((int64_t *)aggregate_result) = std::max(*((int64_t *)aggregate_result), *(int64_t *)batch_aggregate_results[i]);
                        break;
                    case AggregateFunctionType::MIN:
                        // TODO min is not working in date
                        *((int64_t *)aggregate_result) = std::min(*((int64_t *)aggregate_result), *(int64_t *)batch_aggregate_results[i]);
                        break;
                    default:
                        throw "Unsupported aggregate function for date column: ";
                    }
                    break;
                }
            }
        }
        for (auto device_struct_ptr : device_struct_ptrs)
        {
            DeviceStruct::deleteStruct(device_struct_ptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        timeSum += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    std::cout << "Total time taken by aggregate: " << timeSum << " milliseconds" << std::endl;
    std::cout << "================" << std::endl;
    for (auto &[column_name, aggregate_functions_column] : column_aggregate_functions)
    {
        for (int i = 0; i < column_aggregate_results[column_name].size(); i++)
        {
            switch (table->getColumnType(column_name))
            {
            case ColumnType::FLOAT:
                std::cout << "Float aggregate result: " << *((float *)column_aggregate_results[column_name][i]) << std::endl;
                break;
            case ColumnType::DATE:
                std::cout << "Date aggregate result: " << *((int64_t *)column_aggregate_results[column_name][i]) << std::endl;
                break;
            }
        }
    }
    std::cout << "================" << std::endl;
    std::cout << "Total time taken by aggregate only: " << timeSum << " milliseconds" << std::endl;
    // std::cout << "Copy time: " << copy_time << " milliseconds" << std::endl;
    // std::cout << "Kernel time: " << kernel_time << " milliseconds" << std::endl;
    std::vector<void *> results;
    for (int i = 0; i < column_names.size(); i++)
    {
        int idx = 0;
        switch (aggregate_functions[i])
        {
        case AggregateFunctionType::SUM:
        {
            idx = std::find(column_aggregate_functions[column_names[i]].begin(), column_aggregate_functions[column_names[i]].end(), AggregateFunctionType::SUM) - column_aggregate_functions[column_names[i]].begin();
            void *originalValue = column_aggregate_results[column_names[i]][idx];
            float *newValue = new float(*(float *)originalValue);
            results.push_back((void *)newValue);
            break;
        }

        case AggregateFunctionType::MAX:
        {
            idx = std::find(column_aggregate_functions[column_names[i]].begin(), column_aggregate_functions[column_names[i]].end(), AggregateFunctionType::MAX) - column_aggregate_functions[column_names[i]].begin();
            results.push_back(column_aggregate_results[column_names[i]][idx]);
            break;
        }
        case AggregateFunctionType::MIN:
        {
            idx = std::find(column_aggregate_functions[column_names[i]].begin(), column_aggregate_functions[column_names[i]].end(), AggregateFunctionType::MIN) - column_aggregate_functions[column_names[i]].begin();
            results.push_back(column_aggregate_results[column_names[i]][idx]);
            break;
        }
        default:
            throw "Unsupported aggregate function: " + column_names[i];
        }
    }
    return results;
}

void aggregateCPU(std::shared_ptr<Table> &table, std::vector<AggregateFunctionType> &aggregate_functions, std::vector<std::string> &column_names)
{
    auto total_time = 0;
    std::cout<<"Using CPU aggregate"<<std::endl;
    while (table->hasMoreData())
    {
        table->readNextBatch();
        std::vector<std::shared_ptr<ColumnBatch>> current_batch = table->getCurrentBatch();
        size_t num_rows = current_batch[0]->getNumRows();
        std::cout << "Num rows: " << num_rows << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < column_names.size(); i++)
        {
            switch (aggregate_functions[i])
            {
            case AggregateFunctionType::SUM:
            {
                size_t projected_index = table->getColumnIndexProjected(column_names[i]);
                size_t num_rows = current_batch[projected_index]->getNumRows();
                std::shared_ptr<ColumnBatch> column_batch = current_batch[projected_index];
                float sum = 0;
                for (size_t j = 0; j < num_rows; j++)
                {
                    sum += column_batch->getDouble(j);
                }
                std::cout << "Sum of column " << column_names[i] << ": " << sum << std::endl;
                break;
            }
            case AggregateFunctionType::AVG:
            {
                size_t projected_index = table->getColumnIndexProjected(column_names[i]);
                size_t num_rows_avg = current_batch[projected_index]->getNumRows();
                std::shared_ptr<ColumnBatch> column_batch_avg = current_batch[projected_index];
                float sum_avg = 0;
                for (size_t j = 0; j < num_rows_avg; j++)
                {
                    sum_avg += column_batch_avg->getDouble(j);
                }
                std::cout << "Avg of column " << column_names[i] << ": " << sum_avg / num_rows_avg << std::endl;
                break;
            }
            case AggregateFunctionType::COUNT:
            {
                size_t projected_index = table->getColumnIndexProjected(column_names[i]);
                size_t num_rows_count = current_batch[projected_index]->getNumRows();
                std::cout << "Count of column " << column_names[i] << ": " << num_rows_count << std::endl;
                break;
            }
            case AggregateFunctionType::MAX:
            {
                switch (table->getColumnType(column_names[i]))
                {
                case ColumnType::FLOAT:
                {
                    size_t projected_index = table->getColumnIndexProjected(column_names[i]);
                    size_t num_rows_max = current_batch[projected_index]->getNumRows();
                    std::shared_ptr<ColumnBatch> column_batch_max = current_batch[projected_index];
                    float max_value = column_batch_max->getDouble(0);
                    for (size_t j = 1; j < num_rows_max; j++)
                    {
                        max_value = std::max(max_value, column_batch_max->getDouble(j));
                    }
                    std::cout << "Max of column " << column_names[i] << ": " << max_value << std::endl;
                    break;
                }
                case ColumnType::DATE:
                {
                    size_t projected_index = table->getColumnIndexProjected(column_names[i]); 
                    size_t num_rows_max_date = current_batch[projected_index]->getNumRows();
                    std::shared_ptr<ColumnBatch> column_batch_max_date = current_batch[projected_index];
                    int64_t max_value_date = column_batch_max_date->getDateAsInt64(0);
                    for (size_t j = 1; j < num_rows_max_date; j++)
                    {
                        max_value_date = std::max(max_value_date, column_batch_max_date->getDateAsInt64(j));
                    }
                    std::cout << "Max of column " << column_names[i] << ": " << max_value_date << std::endl;
                    break;
                }
                default:
                    std::cout << "Unsupported column type for MAX: " << column_names[i] << std::endl;
                    break;
                }
                break;
            }
            case AggregateFunctionType::MIN:
            {
                switch (table->getColumnType(column_names[i]))
                {
                case ColumnType::FLOAT:
                {
                    size_t projected_index = table->getColumnIndexProjected(column_names[i]);
                    size_t num_rows_min = current_batch[projected_index]->getNumRows();
                    std::shared_ptr<ColumnBatch> column_batch_min = current_batch[projected_index];
                    float min_value = column_batch_min->getDouble(0);
                    for (size_t j = 1; j < num_rows_min; j++)
                    {
                        min_value = std::min(min_value, column_batch_min->getDouble(j));
                    }
                    std::cout << "Min of column " << column_names[i] << ": " << min_value << std::endl;
                    break;
                }
                case ColumnType::DATE:
                {
                    size_t projected_index = table->getColumnIndexProjected(column_names[i]);
                    size_t num_rows_min_date = current_batch[projected_index]->getNumRows();
                    std::shared_ptr<ColumnBatch> column_batch_min_date = current_batch[projected_index];
                    int64_t min_value_date = column_batch_min_date->getDateAsInt64(0);
                    for (size_t j = 1; j < num_rows_min_date; j++)
                    {
                        min_value_date = std::min(min_value_date, column_batch_min_date->getDateAsInt64(j));
                    }
                    std::cout << "Min of column " << column_names[i] << ": " << min_value_date << std::endl;
                    break;
                }
                default:
                    std::cout << "Unsupported column type for MIN: " << column_names[i] << std::endl;
                    break;
                }
                break;
            }
            default:
                std::cout << "Unsupported aggregate function: " << aggregateFunctionTypeToString(aggregate_functions[i]) << std::endl;
                break;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    std::cout << "Total time taken by CPU aggregate: " << total_time << " milliseconds" << std::endl;
    std::cout << "================" << std::endl;
}
