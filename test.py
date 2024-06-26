import torch
import time
from collections import deque, defaultdict, OrderedDict

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

# print(device_modules)
# print(offloaded_modules)

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

