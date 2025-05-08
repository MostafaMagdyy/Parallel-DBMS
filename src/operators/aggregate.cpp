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
    
    //TODO add avg by using sum and count of rows by batches
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
    
        std::vector<std::shared_ptr<ColumnBatch>> current_batch = table->getCurrentBatch();
        for (auto &[column_name, aggregate_functions_column] : column_aggregate_functions)
        {
            std::vector<DeviceStruct> device_struct_ptrs;
            std::vector<AggregateFunctionType> aggregate_functions;
            // TODO: create device struct for this column
            size_t projected_index = table->getColumnIndexProjected(column_name);
            std::shared_ptr<ColumnBatch> column_batch = current_batch[projected_index];
    
            auto start = std::chrono::high_resolution_clock::now();
            column_batch->transferToGPU();
            auto end = std::chrono::high_resolution_clock::now();
            copy_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            DeviceStruct *device_struct_ptr = column_batch->getCpuStructPtr();
            for (auto &aggregate_function : aggregate_functions_column)
            {
                device_struct_ptrs.push_back(*device_struct_ptr);
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
            DeviceStruct::deleteStruct(device_struct_ptrs[0]);
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
    for(int i = 0; i < column_names.size(); i++) {
        int idx = 0;
        switch (aggregate_functions[i])
        {
        case AggregateFunctionType::SUM:
            idx = std::find(column_aggregate_functions[column_names[i]].begin(), column_aggregate_functions[column_names[i]].end(), AggregateFunctionType::SUM) - column_aggregate_functions[column_names[i]].begin();
            results.push_back(column_aggregate_results[column_names[i]][idx]);
            break;
        case AggregateFunctionType::MAX:
            idx = std::find(column_aggregate_functions[column_names[i]].begin(), column_aggregate_functions[column_names[i]].end(), AggregateFunctionType::MAX) - column_aggregate_functions[column_names[i]].begin();
            results.push_back(column_aggregate_results[column_names[i]][idx]);
            break;
        case AggregateFunctionType::MIN:
            idx = std::find(column_aggregate_functions[column_names[i]].begin(), column_aggregate_functions[column_names[i]].end(), AggregateFunctionType::MIN) - column_aggregate_functions[column_names[i]].begin();
            results.push_back(column_aggregate_results[column_names[i]][idx]);
            break;
        default:    
            throw "Unsupported aggregate function: " + column_names[i];
        }
    }
    return results;
}
