import torch
import time
from collections import deque, defaultdict, OrderedDict

# 0. 假设一共有2层，每层2个expert，dim为2
# device_modules = torch.randn(2,2).cuda()
# offloaded_modules = torch.randn(4,2).cuda()
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
device_modules = torch.randn(32,720).cuda()
offloaded_modules = torch.randn(64,720).cuda()
for i in range(32):
    device_modules[i] = offloaded_modules[i]*1.0
offloaded_modules = offloaded_modules.cpu()
experts_info = torch.arange(32).cuda()
selected_experts = torch.arange(32).cuda()
layer_id = 1
experts_list = torch.zeros(32).cuda().to(torch.int64)
experts_prefer_order = torch.arange(32).cuda()

print(device_modules)
print(offloaded_modules)

def load_experts(device_modules,offloaded_modules,experts_info,selected_experts,experts_prefer_order,layer_id,experts_list):
    num_bytes = (device_modules.numel() * device_modules.element_size()) // device_modules.shape[0]
    expert_num = experts_list.shape[0]
    
    offloaded_modules = offloaded_modules.cuda()
    experts_info_list = experts_info.tolist()
    selected_experts = selected_experts.tolist()
    experts_prefer_order_list = experts_prefer_order.tolist()
    experts_prefer_order_dict = OrderedDict()
    #创建orderdict，给preferOrder打顺序标签
    for item in experts_prefer_order_list:
        experts_prefer_order_dict[item] = item

    #把select的放到最后
    for i, exp_info in enumerate(experts_info_list):
        exp_idx_tmp = exp_info - layer_id * expert_num 
        if exp_idx_tmp in selected_experts:
            experts_prefer_order_dict.move_to_end(i,last=True)
    
    #选择最不重要的expert，输出是在device_modules里面的位置
    def choose_expert_to_evict():
        for pos_id, _ in experts_prefer_order_dict.items():
            return pos_id
    
    #遍历选择的expert，如果已经在device_modules则直接写进experts_list，不在就从offloaded_modules load，并更新experts_prefer_order，experts_info和experts_list
    for count, expert_id in enumerate(selected_experts):
        uid = layer_id * expert_num + expert_id #总id
        flag = False
        for  i, loaded_uid in enumerate(experts_info_list):
            if uid == loaded_uid:
                flag = True
                pos_id = i
                break
        if not flag :
            pos_id = choose_expert_to_evict()
            device_modules[pos_id] = offloaded_modules[uid]
            experts_info[pos_id] = uid
            experts_info_list[pos_id]=uid
            experts_prefer_order_dict.move_to_end(pos_id,last=True)
        
        experts_list[count]=pos_id * num_bytes
    
    count=0
    for pos_id, _ in experts_prefer_order_dict.items():
        experts_prefer_order[count] = pos_id
        count+=1

num_iterations = 100
total_time_demo = 0

for _ in range(num_iterations):
    start_time = time.perf_counter()
    load_experts(device_modules,offloaded_modules,experts_info,selected_experts,experts_prefer_order,layer_id,experts_list)
    end_time = time.perf_counter()
    total_time_demo += end_time - start_time

average_time_demo = total_time_demo / num_iterations

print(device_modules)
print(experts_info)
print(experts_prefer_order)
print(experts_list)
print("average_time_demo:",average_time_demo)

