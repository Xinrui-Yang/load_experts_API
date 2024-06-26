#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <string.h>

#define BLOCK_SIZE 128

/* 1. 把需要load的experts存到unloaded数组中，并使用reduction和prefix sum计算需要load的数量。 */
__global__ void load_experts_kernel(
    float *device_modules,
    long *experts_info,
    long *experts_list,
    long *selected_experts,
    int layer_id,
    int token_num,
    int topk,
    int device_num,
    int *unloaded,
    int *block_num,
    int single_sel_num,
    int num_bytes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_id = 0;

    typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int thread_data = 0;

    if (idx < single_sel_num)
    {
        int flag = 0;
        for (int k = 0; k < device_num; ++k)
        {
            if (selected_experts[idx] + layer_id * device_num == experts_info[k])
            {
                pos_id = k;
                flag = 1;
                break;
            }
        }
        thread_data = (flag == 0);
        unloaded[idx] = selected_experts[idx] * (flag == 0) + (flag == 1) * (-1);

        experts_list[idx] = pos_id * num_bytes;
    }

    int aggregate = BlockReduce(temp_storage).Sum(thread_data);
    if (threadIdx.x == 0)
    {
        block_num[blockIdx.x] = aggregate;
    }
}

/* 2. 根据需要load的数量，从前到后选取experts_prefer_order中元素作为pos_id。最后再对experts_prefer_order进行排序。 */
__global__ void load_experts_list_kernel(
    float *device_modules,
    float *offloaded_modules,
    long *experts_info,
    long *experts_prefer_order,
    long *tmp_experts_prefer_order,
    long *experts_list,
    int *unloaded,
    int *unloaded_num,
    int layer_id,
    int single_sel_num,
    int device_num,
    int grid_size,
    int num_bytes,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long pos_id = 0;

    int important_experts = device_num - unloaded_num[grid_size] + idx;
    int less_important_experts = idx + unloaded_num[grid_size];

    if (idx < unloaded_num[grid_size])
    {
        pos_id = experts_prefer_order[idx];
        int j = -1;
        for (int i = 0; i < idx + 1; i++)
        {
            j++;
            while (unloaded[j] == -1)
            {
                j++;
            }
        }
        for(int i = 0; i < dim; ++i){
            device_modules[dim * pos_id + i] = offloaded_modules[dim * (unloaded[j] + layer_id * device_num) + i];
        }
        experts_info[pos_id] = unloaded[j] + layer_id * device_num;
        experts_list[j] = pos_id * num_bytes;
    }
    if (less_important_experts < device_num)
    {
        tmp_experts_prefer_order[less_important_experts] = experts_prefer_order[less_important_experts];
        experts_prefer_order[idx] = tmp_experts_prefer_order[less_important_experts];
    }
    if (important_experts < device_num)
    {
        experts_prefer_order[important_experts] = pos_id;
    }
}

void load_experts_cuda(
    float *device_modules,
    float *offloaded_modules,
    long *experts_info,
    long *selected_experts,
    long *experts_prefer_order,
    int layer_id,
    long *experts_list,
    int offloaded_num,
    int token_num,
    int topk,
    int device_num,
    int dim)
{
    int grid_size = (offloaded_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_bytes = dim * sizeof(device_modules[0]);
    dim3 grid(grid_size);
    dim3 block(BLOCK_SIZE);

    float *d_offloaded_modules = nullptr;
    int *d_unloaded = nullptr;
    int *unloaded_num = nullptr;
    int *block_num = nullptr;
    long *tmp_experts_prefer_order = nullptr;

    cudaMalloc((void **)&d_offloaded_modules, dim * offloaded_num * sizeof(offloaded_modules[0]));
    cudaMemcpy(d_offloaded_modules, offloaded_modules, dim * offloaded_num * sizeof(offloaded_modules[0]), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_unloaded, offloaded_num * sizeof(int));
    cudaMalloc((void **)&unloaded_num, (grid_size + 1) * sizeof(int));
    cudaMalloc((void **)&block_num, grid_size * sizeof(int));
    cudaMalloc((void **)&tmp_experts_prefer_order, device_num * sizeof(long));

    // cudaMallocManaged((void **)&d_offloaded_modules, dim * offloaded_num * sizeof(offloaded_modules[0]));
    // memcpy(d_offloaded_modules, offloaded_modules, dim * offloaded_num * sizeof(offloaded_modules[0]));
    
    thrust::device_ptr<long> d_selected_experts(selected_experts);
    thrust::device_vector<long> d_vec_selected_experts(d_selected_experts, d_selected_experts + token_num * topk);
    thrust::sort(d_vec_selected_experts.begin(), d_vec_selected_experts.end());
    thrust::device_vector<long>::iterator new_end = thrust::unique(d_vec_selected_experts.begin(), d_vec_selected_experts.end());
    int single_sel_num = new_end - d_vec_selected_experts.begin();
    d_vec_selected_experts.resize(single_sel_num);

    /* 1. 把需要load的experts存到unloaded数组中，并使用reduction计算每个block需要load的experts数量。 */
    load_experts_kernel<<<grid, block>>>(
        device_modules,
        experts_info,
        experts_list,
        selected_experts,
        layer_id,
        token_num,
        topk,
        device_num,
        d_unloaded,
        block_num,
        single_sel_num,
        num_bytes);

    /* 2. 使用parallel prefix sums计算所有需要load的experts数量。 */
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        block_num, unloaded_num, grid_size + 1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        block_num, unloaded_num, grid_size + 1);

    /* 3. 根据需要load的数量，从前到后选取experts_prefer_order中元素作为pos_id。最后再对experts_prefer_order进行排序。 */
    load_experts_list_kernel<<<grid, block>>>(
        device_modules,
        d_offloaded_modules,
        experts_info,
        experts_prefer_order,
        tmp_experts_prefer_order,
        experts_list,
        d_unloaded,
        unloaded_num,
        layer_id,
        single_sel_num,
        device_num,
        grid_size,
        num_bytes,
        dim);

    cudaFree(d_offloaded_modules);
    cudaFree(d_unloaded);
    cudaFree(unloaded_num);
    cudaFree(block_num);
    cudaFree(tmp_experts_prefer_order);

}
