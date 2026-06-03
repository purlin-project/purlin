import os

import torch
import torch.distributed as dist

import suture

def main(device: torch.device):
    stream = torch.cuda.current_stream()
    s = stream.cuda_stream
    handle = suture.initialize(dist.group.WORLD, device, 80, s)
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16, device=device)
    t_out = torch.empty_like(t)
    suture.all_reduce(t, t_out, handle, s)
    suture.finalize(handle, s)
    stream.synchronize()
    print(t_out)

def init_pg(device: torch.device):
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=int(os.environ["RANK"]),
        world_size=world_size,
        device_id=device
    )

if __name__ == "__main__":
    if not os.environ.get("LOCAL_RANK"):
        exit(1)
    local_rank = int(os.environ["LOCAL_RANK"])
    device_ = torch.device("cuda", local_rank)
    init_pg(device_)
    main(device_)
    dist.destroy_process_group()