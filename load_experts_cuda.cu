#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>

#define BLOCK_SIZE 256

/* 矩阵去重 */
__device__ void removeDuplicates(long *matrix, int rows, int cols, int result[], int *size)
{
    int *uniqueElements = new int[rows * cols];
    int count = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int element = matrix[i * cols + j];
            int found = 0;
            for (int k = 0; k < count; k++)
            {
                if (uniqueElements[k] == element)
                {
                    found = 1;
                    break;
                }
            }
            if (!found)
            {
                uniqueElements[count++] = element;
            }
        }
    }
    for (int i = 0; i < count; i++)
    {
        result[i] = uniqueElements[i];
    }

    *size = count;
}

/* 1. 把需要load的experts存到unloaded数组中，并使用reduction和prefix sum计算需要load的数量。 */
__global__ void load_experts_kernel(
    float *device_modules,
    float *offloaded_modules,
    long *experts_info,
    long *experts_prefer_order,
    long *experts_list,
    long *selected_experts,
    int layer_id,
    int token_num,
    int topk,
    int device_num,
    int *unloaded,
    int *expert_num_cal,
    int offloaded_num,
    int *single_sel,
    int *block_num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_id = 0, single_sel_num = 0;
    int num_bytes = 2 * sizeof(device_modules[0]);

    removeDuplicates(selected_experts, token_num, topk, single_sel, &single_sel_num);
    int expert_num = single_sel_num;
    *expert_num_cal = single_sel_num;

    typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int thread_data = 0;

    if (idx < single_sel_num)
    {
        int flag = 0;
        for (int k = 0; k < device_num; ++k)
        {
            if (single_sel[idx] + layer_id * expert_num == experts_info[k])
            {
                pos_id = k;
                flag = 1;
                break;
            }
        }
        thread_data = (flag == 0);
        unloaded[idx] = single_sel[idx] * (flag == 0) + (flag == 1) * (-1);

        experts_list[idx] = pos_id * num_bytes;
    }

    int aggregate = BlockReduce(temp_storage).Sum(thread_data);
    if (threadIdx.x == 0)
    {
        block_num[blockIdx.x] = aggregate;
    }
}

__global__ void prefix_sum(int *src, int *dst, int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        dst[0] = 0;
        for (int i = 1; i < n; i++)
        {
            dst[i] = dst[i - 1] + src[i - 1];
        }
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
    int *expert_num,
    int device_num,
    int grid_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long pos_id = 0;
    int num_bytes = 2 * sizeof(device_modules[0]);

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
        device_modules[2 * pos_id] = offloaded_modules[2 * (unloaded[j] + layer_id * (*expert_num))];
        device_modules[2 * pos_id + 1] = offloaded_modules[2 * (unloaded[j] + layer_id * (*expert_num)) + 1];
        experts_info[pos_id] = unloaded[j] + layer_id * (*expert_num);
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
    int device_num)
{
    int grid_size = (offloaded_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_size);
    dim3 block(BLOCK_SIZE);

    float *d_offloaded_modules = nullptr;
    int *d_unloaded = nullptr;
    int *unloaded_num = nullptr;
    int *expert_num_cal = nullptr;
    int *single_sel = nullptr;
    int *block_num = nullptr;
    long *tmp_experts_prefer_order = nullptr;

    cudaMalloc((void **)&d_offloaded_modules, 2 * offloaded_num * sizeof(offloaded_modules[0]));
    cudaMalloc((void **)&d_unloaded, offloaded_num * sizeof(int));
    cudaMalloc((void **)&unloaded_num, (grid_size + 1) * sizeof(int));
    cudaMalloc((void **)&expert_num_cal, sizeof(int));
    cudaMalloc((void **)&single_sel, offloaded_num * sizeof(int));
    cudaMalloc((void **)&block_num, grid_size * sizeof(int));
    cudaMalloc((void **)&tmp_experts_prefer_order, device_num * sizeof(long));

    cudaMemcpy(d_offloaded_modules, offloaded_modules, 2 * offloaded_num * sizeof(offloaded_modules[0]), cudaMemcpyHostToDevice);

    load_experts_kernel<<<grid, block>>>(
        device_modules,
        d_offloaded_modules,
        experts_info,
        experts_prefer_order,
        experts_list,
        selected_experts,
        layer_id,
        token_num,
        topk,
        device_num,
        d_unloaded,
        expert_num_cal,
        offloaded_num,
        single_sel,
        block_num);

    prefix_sum<<<1, 1>>>(block_num, unloaded_num, grid_size + 1);

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
        expert_num_cal,
        device_num,
        grid_size);

    cudaFree(d_offloaded_modules);
    cudaFree(d_unloaded);
    cudaFree(unloaded_num);
    cudaFree(expert_num_cal);
    cudaFree(single_sel);
    cudaFree(block_num);
    cudaFree(tmp_experts_prefer_order);
}