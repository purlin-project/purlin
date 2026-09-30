# Purlin
**Purlin is a high-performance GPU communication framework for the scale-up domain, such as an NVLink-connected server.** 

Purlin provides fast, single-kernel collectives and reusable device-side primitives for building custom communication or fusing with computation.

Purlin includes AllReduce, AllGather, ReduceScatter, and AllToAll, plus variable-length variants of the latter three. 

Underneath these collectives are two hardware-aware primitives: **copy** and **reduce**, which we also expose via the device-side Atom interface.

See [paper](https://arxiv.org/abs/2609.36954) for more details.

## Key Idea
Our key innovation is **decoupling orchestration from the datapath** of collective communication. That is, separating *where and when* data moves from *how* the GPU moves it. 

- **Layouts describe the collective:** how inputs and outputs are distributed, and whether to copy or reduce.
- **SNAC coordinates execution:** the shared *Stage, Notify, And Consume* (SNAC) protocol derives orchestration from those layouts.
- **Atoms move the data:** hardware-specific implementations perform the copies and reductions as coordinated by SNAC.

This separation makes Purlin **evolvable**: 
- New hardware mechanisms can be added through Atoms without rewriting orchestration 
- Communication is much easier to customize either at the collective level through our layouts or via composing our Atoms. 

Also, Purlin allows for programmability (see [here](https://github.com/purlin-project/purlin/blob/a8b67c70b5aba73a45df56e436c26b6c91ce18dc/csrc/include/purlin/host/allReduce.cuh#L90) for an example and [codesign](csrc/include/purlin/host/codesign.cuh) for more details) to achieve peak performance.


## 🧨 QuickStart
```bash
uv pip install purlin # best to use a venv here
torchrun --nproc-per-node <num-of-gpus> quickstart.py 
```


## Using Purlin 
This documentation is a work-in-progress.
<details>
<summary>CUDA C++</summary>

## Host-side

Create a context once and reuse it for your collectives.

First, set up `workspace` with symmetric staging and signal buffers. 

You can do that via [makePurlinWorkspace](csrc/support/purlin/benchmark/purlin_runtime.cuh) 
or for a more generic but involved way, see [purlin_initialize](purlin/bindings.py).

```cpp
#include <purlin/host.cuh>

auto ctx = purlin::initialize(rank, world, workspace, stream);
purlin::allReduce<ARCH, __nv_bfloat16>(src, dst, bytes, ctx, stream);
purlin::finalize(ctx, stream);
```

`src` and `dst` are regular pointers to device memory.

Set `ARCH` for your GPU, for example `900` for SM90. The collective runs on
`stream`. 

Call `finalize` after your last collective to free internal buffers.

## Device-side

Use `<purlin/core.cuh>` to call an Atom or a collective from inside your kernel.

See [examples](csrc/examples) as a starting point.

For tuning, see [host code](csrc/include/purlin/host), [tuning policies](csrc/include/purlin/host/codesign.cuh).

### Samples

In the example below, `PurlinAtom::THREADS` copy one aligned slice. 

`src` or `dst` pointers can refer to local GPU memory or remote memory on a peer GPU:

```cpp
#include <purlin/core.cuh>

template<typename PurlinAtom>
__global__ void copyKernel(cuda::std::byte* dst, const cuda::std::byte* src,
                          size_t bytesPerBlock) {
  extern __shared__ __align__(128) cuda::std::byte workspace[];
  const auto offset = blockIdx.x * bytesPerBlock;
  PurlinAtom::copy(dst + offset, src + offset, bytesPerBlock, workspace);
}
```

Launch it with:

- `PurlinAtom::THREADS` threads per block.
- `PurlinAtom::COPY_PIPELINE_SMEM_BYTES` bytes of dynamic shared memory.

See the [point-to-point example](csrc/examples/p2p/push.cu) for a full example.

To call a collective, pass the context you created on the host like below


```cpp
#include <purlin/core.cuh>
template<typename PurlinAtom, typename CollConfig>
__global__ void allGatherKernel(const cuda::std::byte* src, cuda::std::byte* dst,
                            size_t bytesPerRank, const __grid_constant__ purlin::Context ctx) {
  extern __shared__ __align__(128) cuda::std::byte workspace[];
  // do some work
  const int blocks = static_cast<int>(gridDim.x);
  const purlin::SnacArgs args{
    .dst = dst,
    .src = src,
    .bytes = bytesPerRank,
    .workspace = workspace,
    .blocks = blocks,
    .collBlocks = blocks,
    .bIdx = static_cast<int>(blockIdx.x)
  };
  purlin::allGather<PurlinAtom, CollConfig>(args, ctx);
  // do some other work
}
```

Use `PurlinAtom::THREADS` threads per block. See [here](csrc/examples/ag/ag.cu) for a more complete example

Exactly `args.blocks` CTAs must enter the collective, where each CTA has `PurlinAtom::THREADS` threads.

</details>

<details>
<summary>Python</summary>

First, initialize `torch.distributed` and select the local CUDA `device`.

Here, `x` is a contiguous CUDA tensor of float32 values. Create an output tensor,
then call AllReduce:

```python
import torch
import torch.distributed as dist
import purlin

stream = torch.cuda.current_stream(device)
major, minor = torch.cuda.get_device_capability(device)
handle = purlin.initialize(
    dist.group.WORLD, device, major * 10 + minor, stream.cuda_stream
)
out = torch.empty_like(x)
purlin.all_reduce(x, out, handle, stream.cuda_stream)
purlin.finalize(handle, stream.cuda_stream)
```

The first `initialize` call compiles the CUDA code and caches it for reuse.
See [quickstart.py](quickstart.py) for a complete script.

</details>

Note: 

- Keep buffer addresses and byte counts aligned for your Atom. We use 32-byte alignment for Blackwell and 16 for others.

- For variable-length collectives, follow the setup in
[bindings.py](purlin/bindings.py) and the [host code](csrc/include/purlin/host).


## C++ benchmarks
<details>
<summary>Click here to see steps</summary>

Dependencies:

- CUDA Toolkit
- [NVSHMEM](https://developer.nvidia.com/nvshmem-downloads?target_os=Linux) (for C++ symmetric memory)
- MPI (to launch processes)
- CMake 3.27+
- Ninja
- [CPM](https://github.com/cpm-cmake/cpm.cmake#adding-cpm) (for CMake dependency management)

Set `NVSHMEM_LIB_HOME` to your NVSHMEM library directory.
From the repository root:

```bash
export NVSHMEM_LIB_HOME=/path/to/nvshmem/lib
cmake -S csrc -B csrc/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build csrc/build --target testAR testA2A testA2AV testAG testAGV testRS testRSV
```

| Collective | Binary |
| --- | --- |
| AllReduce | `testAR` |
| AllToAll / AllToAllV | `testA2A` / `testA2AV` |
| AllGather / AllGatherV | `testAG` / `testAGV` |
| ReduceScatter / ReduceScatterV | `testRS` / `testRSV` |

> 📏 **Understanding `totalBytes`**
>
> The reported `totalBytes` is what we use to compute bandwidth. 
> It is the output size for AllGather.
> For the other collectives, it is the input size per rank (also the output size
> for AllReduce and AlltoAll).

Run with one MPI process per GPU; for example, on 8 GPUs:

```bash
# each rank gets as input 1K...1G with output also the same size for the below
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none mpirun -n 8 ./csrc/build/testAR 1K 1G
# each rank gets as input: 128...128M but output is multiplied by world for allGather so 1K...1G  
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none mpirun -n 8 ./csrc/build/testAG 128 128M
```

Arguments: `[minBytes] [maxBytes] [graphLaunches] [runs] [warmup] [seed]`.

</details>


## Implementation Details
Below are some details which you may likely be curious about after reading the paper.

## Collective Namings (using Layouts)
see [here](csrc/include/purlin/collective.cuh).

## SNAC Details
see [here](csrc/include/purlin/snac.cuh). Note this is over 2K LOC.

## Atom implementations
<details>
<summary>Click to see linked files</summary>


| Atom      | File                                     |
|-----------|------------------------------------------|
| `Generic` | [here](csrc/include/purlin/fascia.cuh)   |
| `Ampere`  | [here](csrc/include/purlin/tendon.cuh)   |
| `Hopper`  | [here](csrc/include/purlin/ligament.cuh) |
| `Blackwell` | [here](csrc/include/purlin/cortex.cuh)    |

</details>

## Codesign Policies
see [here](csrc/include/purlin/host/codesign.cuh)


## Deterministic Reduction
<details>
<summary>Click to see details</summary>

Enable deterministic mode for bit-wise repeatable results with the same inputs and operator.

> 👉 This configuration only affects Hopper and above because non-deterministic mode
*allows* NVLS, while deterministic mode *disables* it. 
> 
> For Ampere and below, both
modes use the deterministic path *always* which gives peak performance there.

All ranks must use the same mode.

C++:

```cpp
// this call uses deterministic mode
purlin::allReduce<..., purlin::ReductionMode::deterministic>(src, dst, bytes, ctx, stream);
// this call uses the default mode
purlin::allReduce<...>(src, dst, bytes, ctx, stream);
```

Python:

```python
purlin.all_reduce(..., reduction_mode=purlin.ReductionMode.DETERMINISTIC)
```

Default is non-deterministic.

</details>
