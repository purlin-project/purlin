# Purlin
**Purlin is a high-performance GPU communication framework for moving and reducing data across GPUs within a scale-up domain, such as an NVLink-connected server.** It provides both ready-to-use, single-kernel collectives and reusable device-side primitives for building custom communication and fusing it with computation.

Purlin includes AllReduce, AllGather, ReduceScatter, and AllToAll, plus variable-length variants of the latter three. Underneath these collectives are two hardware-aware primitives: **copy** and **N-to-1 reduce**, which combines multiple input buffers into one output.

## Key Idea
The key idea is **decoupling orchestration from the datapath**: separating *where and when* data moves from *how* the GPU moves it. 

- **Layouts describe the collective:** how inputs and outputs are distributed across GPUs, and whether contributions are copied or reduced.
- **SNAC coordinates execution:** the shared *Stage, Notify, And Consume* protocol derives communication and synchronization from those layouts, ensuring data is ready before consumption and buffers are safe to reuse.
- **Atoms move the data:** hardware-specific implementations perform the copies and reductions using mechanisms suited to each GPU generation.

This separation makes Purlin **evolvable**: new hardware mechanisms can be added through Atoms without rewriting collective orchestration, and new collective variants can be expressed through layouts while reusing SNAC. Hardware-specific tuning preserves performance across latency-sensitive and bandwidth-intensive workloads, with implementations for Ampere, Hopper, and Blackwell GPUs.

## 🧨 QuickStart
```bash
uv pip install purlin # best to use a venv here
torchrun --nproc-per-node <num-of-gpus> quickstart.py 
```

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

## Atom implementations

| Atom | File |
| --- | --- |
| `Atom<700>` | [fascia.cuh](csrc/include/purlin/fascia.cuh) |
| `Atom<800>` | [tendon.cuh](csrc/include/purlin/tendon.cuh) |
| `Atom<900>` | [ligament.cuh](csrc/include/purlin/ligament.cuh) |
| `Atom<1000>` | [cortex.cuh](csrc/include/purlin/cortex.cuh) |
### Reduction determinism

Host reductions default to `purlin::ReductionMode::nonDeterministic`, which allows
multimem when the existing dispatch checks permit it. Select deterministic mode
with the template parameter after the reduction operator:

```cpp
purlin::allReduce<ARCH, float, purlin::ReduceOp::add,
  purlin::ReductionMode::deterministic>(src, dst, bytes, ctx, stream);
```

`reduceScatter` and `reduceScatterV` accept the same parameter. Deterministic mode
skips multicast dispatch and uses the existing unicast, rank-ordered reduction.
On SM90 and newer, that throughput reduction delegates to the `Atom<800>` pipeline.
The guarantee is repeatability for the same inputs, operator, and collective
configuration; non-deterministic mode may produce the same result without that
guarantee.

The Python `all_reduce`, `reduce_scatter`, and `reduce_scatter_v` functions accept
an optional `reduction_mode`, defaulting to `ReductionMode.NON_DETERMINISTIC`:

```python
purlin.all_reduce(src, dst, handle, stream_ptr,
                  reduction_mode=purlin.ReductionMode.DETERMINISTIC)
```

All ranks participating in a collective must select the same mode.
