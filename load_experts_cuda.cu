#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <map>

__device__ int removeDuplicates(int **matrix, int *result) {
    int rows = sizeof(matrix) / sizeof(matrix[0]);
    int cols = sizeof(matrix[0]) / sizeof(matrix[0][0]);

    int uniqueElements[rows * cols]; 
    int count = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int element = matrix[i][j];
            int found = 0;

            for (int k = 0; k < count; k++) {
                if (uniqueElements[k] == element) {
                    found = 1;
                    break;
                }
            }

            if (!found) {
                uniqueElements[count++] = element;
            }
        }
    }

    for (int i = 0; i < count; i++) {
        result[i] = uniqueElements[i];
    }

    return count;
}

__global__ void load_experts_kernel(
    float *device_modules,
    float *offloaded_modules,
    float *experts_info,
    float *experts_prefer_order,
    int layer_id,
    float *experts_list,
    int **selected_experts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_id = 0;
    int expert_num = sizeof(experts_list) / sizeof(experts_list[0]);
    int device_num = sizeof(device_modules) / sizeof(device_modules[0]);
    int num_bytes = sizeof(device_modules[0]);
    int *single_sel;
    int single_sel_num = removeDuplicates(selected_experts,single_sel);
    
    __shared__ int orderId = 0;

    for (int idx = 0; idx < single_sel_num; ++idx){
        for (int k = 0; k < device_num; ++k){
            if (single_sel[idx] + layer_id * expert_num == experts_info[k]){
                pos_id = idx;
            }else{
                __syncthreads();
                pos_id = experts_prefer_order[orderId];
                orderId++;
                device_modules[pos_id] = offloaded_modules[single_sel[idx] + layer_id * expert_num];
                experts_info[pos_id] = single_sel[idx] + layer_id * expert_num;
                // experts_prefer_order.move_to_end(pos_id,last=True)?
            }
        }
        experts_list[idx] = pos_id * num_bytes;
    }
}

void load_experts_cuda(
    float *device_modules,
    float *offloaded_modules,
    float *experts_info,
    int **selected_experts,
    float *experts_prefer_order,
    int layer_id,
    float *experts_list,
    
    int single_sel_num)
{
    float *d_offloaded_modules;
    cudaMalloc((void**)&offloaded_modules, sizeof(offloaded_modules));
    cudaMemcpy(d_offloaded_modules, offloaded_modules, 2*offloaded_num*sizeof(float), cudaMemcpyHostToDevice);

    load_experts_kernel<<<ceil(single_sel_num/256.0),256>>>(
        device_modules,
        offloaded_modules,
        experts_info,
        experts_prefer_order,
        layer_id,
        experts_list,
        selected_experts
        );





    //cudaMemcpy(C, d_C, rows_A*cols_B*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_offloaded_modules);
}