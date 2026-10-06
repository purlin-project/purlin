# Symmetric memory providers

Purlin allocates the workspace through a symmetric memory provider that you choose.

## NVSHMEM convenience provider

The optional [symm_mem.cuh](symm_mem.cuh)
header supplies `NvshmemMemory`. It lives outside the core `include` directory.
Add `<repo>/csrc` to your include paths, include it explicitly, and link NVSHMEM
when using this provider. Neither `core.cuh` nor `host.cuh` includes it, and the
core CMake target does not add its include path.

```cpp
#include <purlin/host.cuh>
#include <contrib/symm_mem.cuh>

nvshmem_init();
const int rank = nvshmem_my_pe();
const int world = nvshmem_n_pes();
CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
cudaStream_t stream;
CHECK_CUDA(cudaStreamCreate(&stream));

auto managed = purlin::initialize(rank, world, stream, purlin::NvshmemMemory{});
auto& ctx = managed.context();
purlin::allReduce<ARCH, __nv_bfloat16>(src, dst, bytes, ctx, stream);
purlin::finalize(managed, stream);

CHECK_CUDA(cudaStreamDestroy(stream));
nvshmem_finalize();
```

`src` and `dst` are regular device buffers allocated on the selected GPU.
The NVSHMEM provider uses the whole NVSHMEM job as its rank group and requires
direct GPU mappings to every peer. Launch one process per GPU in the same scale-up
domain. The application owns NVSHMEM initialization, device selection, and streams.

Set `ARCH` for your GPU, for example `900` for SM90. The collective runs on
`stream`.

## Managed contexts

`initialize` returns a move-only `ManagedContext<Provider>`. Its `context()` method
borrows the unchanged, trivially copyable `Context` used by host APIs and kernels.
Call `finalize(managed, stream)` after the last collective. It waits for the stream,
frees internal buffers and pointer tables, and releases the provider allocation.
It is safe to call again on an empty handle. References and copies of the borrowed
context become invalid after finalization.

All ranks must initialize and finalize in the same order with matching workspace
settings. Order any work on other streams before the stream passed to `finalize`.
If not explicitly finalized, the handle cleans up at destruction using the
initialization stream, which must still exist. Finalize before shutting down the
provider's runtime. Finalize the managed handle, rather than its borrowed context.

The optional last argument sets the throughput staging size per buffer:

```cpp
auto managed = purlin::initialize(rank, world, stream, provider, 32UL * 1024 * 1024);
```

It defaults to 256 MiB and must be between `MIN_CHUNK_SIZE` and `MAX_STAGING_SIZE`,
inclusive, and a multiple of `MAX_ACCESS_ALIGNMENT`. Purlin sizes the allocation
and latency-buffer offsets accordingly. `PURLIN_DISABLE_MULTIMEM` disables all
multicast use; `PURLIN_DISABLE_MULTIMEM_LR` disables it for latency staging only.

## Custom symmetric memory providers

The generic initializer in [setup.cuh](../include/purlin/setup.cuh) has no
NVSHMEM dependency. A provider implements two methods and defines its allocation
type:

```cpp
struct MyMemoryProvider {
  struct Allocation {
    std::vector<void*> peers;
    void* multicast = nullptr;
    // Add any handles needed to release the allocation.
  };

  Allocation allocate_zeroed(size_t bytes, size_t alignment, cudaStream_t stream);
  void deallocate(Allocation& allocation) noexcept;
};
```

- `allocate_zeroed` provides at least `bytes` per rank with the requested alignment.
  All ranks' buffers must be zeroed and all mappings ready before any call returns.
- `peers` is a host array with one GPU-accessible pointer per Purlin rank, in rank
  order. `peers[rank]` addresses local memory. The allocation type may retain
  provider-specific import or ownership handles.
- `multicast` is null when unavailable. Otherwise, it maps the same allocation
  across exactly the participating ranks.
- `deallocate` releases the allocation and its mappings without throwing. Purlin
  completes local stream work before calling it; the provider coordinates ranks
  before invalidating memory. Provider failures must be handled consistently
  across ranks so that peers do not remain blocked in collective operations.

Pass the provider by value, or move it into `initialize` if it is move-only.
The managed handle retains both the provider and its allocation until cleanup.
Purlin handles workspace layout, device pointer tables, counters, and signals.
See `NvshmemMemory` for a complete provider and the
[collective examples](../examples) for its use.

## Caller-owned workspaces

The existing `initialize(rank, world, workspace, stream, stagingSize)` overload
still accepts `WorkspaceMemory` and returns a plain `Context`.
`finalize(ctx, stream)` frees only Purlin's internal CUDA buffers; the caller
retains responsibility for workspace memory and pointer tables. This path is
used by the [Python bindings](../../purlin/bindings.py).
