import torch.autograd.profiler as profiler
import torch
import load_experts_ext
import time

def decorate_trace_handler(rank):
    def trace_handler(prof):
        if rank in [0]:
            prof.export_chrome_trace("test"+str(rank)+".json")
    return trace_handler

prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        # with_stack=False,
        use_cuda = True,
        profile_memory = True,
        schedule=torch.profiler.schedule(
            wait=5,
            warmup=5,
            active=2),
        on_trace_ready=decorate_trace_handler(0)
    )

device_modules = torch.randn(8,2).cuda()
offloaded_modules = torch.randn(16,2).cuda()
for i in range(8):
    device_modules[i] = offloaded_modules[i]*1.0
offloaded_modules = offloaded_modules.cpu()
experts_info = torch.arange(8).cuda()
selected_experts = torch.arange(8).cuda()
layer_id = 1
experts_list = torch.zeros(8).cuda().to(torch.int64)
experts_prefer_order = torch.arange(8).cuda()

with prof:
    for i in range(20):
        load_experts_ext.load_experts_para(
        device_modules,
        offloaded_modules,
        experts_info,
        selected_experts,
        experts_prefer_order,
        layer_id,
        experts_list)
        prof.step()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))