import os

import torch
import torch.distributed as dist

import purlin

def main(device: torch.device):
    stream = torch.cuda.current_stream()
    s = stream.cuda_stream
    major, minor = torch.cuda.get_device_capability(device)
    arch = major * 10 + minor # GPU architecture that purlin will JIT compile the Atom for
    handle = purlin.initialize(dist.group.WORLD, device, arch, s)
    t = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float16, device=device)
    t_out = torch.empty_like(t)
    purlin.all_reduce(t, t_out, handle, s)
    purlin.finalize(handle, s)
    stream.synchronize()
    if dist.get_rank() == 0:
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