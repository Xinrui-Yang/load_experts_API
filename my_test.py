import torch
import time
import load_experts_ext

# 0. 假设一共有2层，每层2个expert，dim为3
# device_modules = torch.randn(2,3).cuda()
# offloaded_modules = torch.randn(4,3).cuda()
# device_modules[0] = offloaded_modules[0]*1.0
# device_modules[1] = offloaded_modules[3]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.tensor([0,3]).cuda()
# selected_experts = torch.tensor([0,1]).cuda()
# layer_id = 0 
# experts_list = torch.zeros(2).cuda().to(torch.int64)
# experts_prefer_order = torch.tensor([1,0]).cuda()

# 1. 假设一共有2层，每层5个expert，dim为2
# device_modules = torch.randn(5,2).cuda()
# offloaded_modules = torch.randn(10,2).cuda()
# device_modules[0] = offloaded_modules[0]*1.0
# device_modules[1] = offloaded_modules[4]*1.0
# device_modules[2] = offloaded_modules[5]*1.0
# device_modules[3] = offloaded_modules[6]*1.0
# device_modules[4] = offloaded_modules[7]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.tensor([0,4,5,6,7]).cuda()
# selected_experts = torch.tensor([0,1,3,4]).cuda()
# layer_id = 0
# experts_list = torch.zeros(4).cuda().to(torch.int64)
# experts_prefer_order = torch.tensor([2,3,4,0,1]).cuda()

# print("in:")
# print("device_modules:")
# print(device_modules)
# print()
# print("offloaded_modules:")
# print(offloaded_modules)
# print()
# print("experts_info:")
# print(experts_info)
# print()
# print("selected_experts:")
# print(selected_experts)
# print()
# print("experts_prefer_order:")
# print(experts_prefer_order)
# print()
# print("experts_list:")
# print(experts_list)
# print()

num_iterations = 100
total_time_cuda = 0

test_device_num = 32
dim = 720
test_layer_id = 1
test_offloaded_num = test_device_num * (test_layer_id + 1)

for _ in range(num_iterations):
    device_modules = torch.randn(test_device_num,dim).cuda()
    offloaded_modules = torch.randn(test_offloaded_num,dim).cuda()
    for i in range(test_device_num):
        device_modules[i] = offloaded_modules[i]*1.0
    offloaded_modules = offloaded_modules.cpu()
    experts_info = torch.arange(test_device_num).cuda()
    selected_experts = torch.arange(test_device_num).cuda()
    layer_id = test_layer_id 
    experts_list = torch.zeros(test_device_num).cuda().to(torch.int64)
    experts_prefer_order = torch.arange(test_device_num).cuda()

    start_time_cuda = time.perf_counter()

    load_experts_ext.load_experts_para(
    device_modules,
    offloaded_modules,
    experts_info,
    selected_experts,
    experts_prefer_order,
    layer_id,
    experts_list)

    end_time_cuda = time.perf_counter()
    total_time_cuda += end_time_cuda - start_time_cuda

average_time_cuda = total_time_cuda / num_iterations 

print()
print("out:")
print("device_modules:")
print(device_modules)
print()
print("experts_info:")
print(experts_info)
print()
print("experts_prefer_order:")
print(experts_prefer_order)
print()
print("experts_list:")
print(experts_list)
print()
print("average_time_cuda:",average_time_cuda)
