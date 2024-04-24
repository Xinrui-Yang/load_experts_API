import torch
import time
from collections import deque, defaultdict, OrderedDict

device_modules = torch.randn(2,2).cuda()
offloaded_modules = torch.randn(4,2).cuda()

device_modules[0] = offloaded_modules[0]*1.0
device_modules[1] = offloaded_modules[3]*1.0

offloaded_modules = offloaded_modules.cpu()

experts_info = torch.tensor([0,3]).cuda()
selected_experts = torch.tensor([0,1]).cuda()
# selected_experts = torch.tensor([[0,1],[0,1]]).cuda()
layer_id = 0
experts_list = torch.zeros(2).cuda().to(torch.int64)
experts_prefer_order = torch.tensor([1,0]).cuda()

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
total_time = 0

for _ in range(num_iterations):
    start_time = time.perf_counter()
    load_experts(device_modules,offloaded_modules,experts_info,selected_experts,experts_prefer_order,layer_id,experts_list)
    end_time = time.perf_counter()
    total_time += end_time - start_time

average_time = total_time / num_iterations

# start_time_cpu = time.time()
# end_time_cpu = time.time()
# execution_time_cpu = end_time_cpu - end_time_cpu
# 9.575322270393372e-05

print("execution_time_cpu:")
print(average_time)

print(device_modules)
print(experts_info)
print(experts_prefer_order)
print(experts_list)

