#include <torch/extension.h>
#include <torch/torch.h>

using namespace torch;

// CUDA declaration
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
    int dim);

// C++ interface
void load_experts_para(
    Tensor device_modules,
    Tensor offloaded_modules,
    Tensor experts_info,
    Tensor selected_experts,
    Tensor experts_prefer_order,
    int layer_id,
    Tensor experts_list)
{
    int offloaded_num = offloaded_modules.size(0);
    int device_num = device_modules.size(0);
    int dim = offloaded_modules.size(1);

    int token_num = 0;
    int topk = 0;
    if (selected_experts.dim() == 1)
    {
        token_num = 1;
        topk = selected_experts.size(0);
    }
    else
    {
        token_num = selected_experts.size(0);
        topk = selected_experts.size(1);
    }

    // cuda kernel
    load_experts_cuda(
        device_modules.data_ptr<float>(),
        offloaded_modules.data_ptr<float>(),
        experts_info.data_ptr<long>(),
        selected_experts.data_ptr<long>(),
        experts_prefer_order.data_ptr<long>(),
        layer_id,
        experts_list.data_ptr<long>(),
        offloaded_num,
        token_num,
        topk,
        device_num,
        dim);
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("load_experts_para", &load_experts_para, "Load Experts (CUDA)");
}
