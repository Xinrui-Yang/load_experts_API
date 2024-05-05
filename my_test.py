import torch
import time
from collections import deque, defaultdict, OrderedDict
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
device_modules = torch.randn(5,2).cuda()
offloaded_modules = torch.randn(10,2).cuda()
device_modules[0] = offloaded_modules[0]*1.0
device_modules[1] = offloaded_modules[4]*1.0
device_modules[2] = offloaded_modules[5]*1.0
device_modules[3] = offloaded_modules[6]*1.0
device_modules[4] = offloaded_modules[7]*1.0
offloaded_modules = offloaded_modules.cpu()
experts_info = torch.tensor([0,4,5,6,7]).cuda()
selected_experts = torch.tensor([0,1,3,4]).cuda()
layer_id = 0
experts_list = torch.zeros(4).cuda().to(torch.int64)
experts_prefer_order = torch.tensor([2,3,4,0,1]).cuda()

# 2. 假设一共有2层，每层8个expert，dim为2
# device_modules = torch.randn(8,2).cuda()
# offloaded_modules = torch.randn(16,2).cuda()
# for i in range(8):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(8).cuda()
# selected_experts = torch.arange(8).cuda()
# layer_id = 1
# experts_list = torch.zeros(8).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(8).cuda()

# 3. 假设一共有2层，每层64个expert，dim为2
# device_modules = torch.randn(64,2).cuda()
# offloaded_modules = torch.randn(128,2).cuda()
# for i in range(64):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(64).cuda()
# selected_experts = torch.arange(64).cuda()
# layer_id = 1
# experts_list = torch.zeros(64).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(64).cuda()

# 4. 假设一共有2层，每层128个expert，dim为2
# device_modules = torch.randn(128,2).cuda()
# offloaded_modules = torch.randn(256,2).cuda()
# for i in range(128):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(128).cuda()
# selected_experts = torch.arange(128).cuda()
# layer_id = 1 
# experts_list = torch.zeros(128).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(128).cuda()

# 5. 假设一共有2层，每层512个expert，dim为2
# device_modules = torch.randn(512,2).cuda()
# offloaded_modules = torch.randn(1024,2).cuda()
# for i in range(512):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(512).cuda()
# selected_experts = torch.arange(512).cuda()
# layer_id = 1 
# experts_list = torch.zeros(512).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(512).cuda()

#################################################

# 1. 假设一共有2层，每层8个expert，dim为58
# device_modules = torch.randn(8,58).cuda()
# offloaded_modules = torch.randn(16,58).cuda()
# for i in range(8):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(8).cuda()
# selected_experts = torch.arange(8).cuda()
# layer_id = 1
# experts_list = torch.zeros(8).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(8).cuda()

# 2. 假设一共有2层，每层8个expert，dim为256
# device_modules = torch.randn(8,256).cuda()
# offloaded_modules = torch.randn(16,256).cuda()
# for i in range(8):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(8).cuda()
# selected_experts = torch.arange(8).cuda()
# layer_id = 1
# experts_list = torch.zeros(8).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(8).cuda()

# 3. 假设一共有2层，每层8个expert，dim为720
# device_modules = torch.randn(8,720).cuda()
# offloaded_modules = torch.randn(16,720).cuda()
# for i in range(8):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(8).cuda()
# selected_experts = torch.arange(8).cuda()
# layer_id = 1
# experts_list = torch.zeros(8).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(8).cuda()

#################################################

# 1. 假设一共有2层，每层32个expert，dim为58
# device_modules = torch.randn(32,58).cuda()
# offloaded_modules = torch.randn(64,58).cuda()
# for i in range(32):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(32).cuda()
# selected_experts = torch.arange(32).cuda()
# layer_id = 1
# experts_list = torch.zeros(32).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(32).cuda()

# 2. 假设一共有2层，每层32个expert，dim为256
# device_modules = torch.randn(32,256).cuda()
# offloaded_modules = torch.randn(64,256).cuda()
# for i in range(32):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(32).cuda()
# selected_experts = torch.arange(32).cuda()
# layer_id = 1
# experts_list = torch.zeros(32).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(32).cuda()

# 3. 假设一共有2层，每层8个expert，dim为720
# device_modules = torch.randn(32,720).cuda()
# offloaded_modules = torch.randn(64,720).cuda()
# for i in range(32):
#     device_modules[i] = offloaded_modules[i]*1.0
# offloaded_modules = offloaded_modules.cpu()
# experts_info = torch.arange(32).cuda()
# selected_experts = torch.arange(32).cuda()
# layer_id = 1
# experts_list = torch.zeros(32).cuda().to(torch.int64)
# experts_prefer_order = torch.arange(32).cuda()

print("in:")
print("device_modules:")
print(device_modules)
print()
print("offloaded_modules:")
print(offloaded_modules)
print()
print("experts_info:")
print(experts_info)
print()
print("selected_experts:")
print(selected_experts)
print()
print("experts_prefer_order:")
print(experts_prefer_order)
print()
print("experts_list:")
print(experts_list)
print()

num_iterations = 1
total_time_cuda = 0

for _ in range(num_iterations):
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
