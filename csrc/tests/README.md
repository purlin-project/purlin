# Collective performance benchmarks

This directory contains matching Purlin and NCCL-only benchmark suites. The
small files under `tests/` and `tests/nccl/` describe only collective-specific
setup, invocation, and validation. Reusable facilities live under `common/`:

- command-line parsing, size sweeps, stream timing, and CUDA Graph timing;
- deterministic byte and floating-point data and reference generation;
- stream-ordered RAII device buffers, communicator/runtime ownership, and error checking;
- variable-count layouts and NCCL emulations; and
- consistent CSV reporting.

## Purlin targets

| Target | Operation | Correctness reference |
| --- | --- | --- |
| `testAG` | AllGather | NCCL AllGather |
| `testAR` | AllReduce | ordered kernel + MatX |
| `testRS` | ReduceScatter | ordered kernel + MatX |
| `testA2A` | AllToAll | NCCL send/receive |
| `testAGV` | AllGatherV | NCCL grouped broadcast |
| `testA2AV` | AllToAllV | NCCL variable send/receive |
| `testRSV` | ReduceScatterV | ordered kernel + MatX |

## NCCL-only targets

| Target | Operation | NCCL implementation |
| --- | --- | --- |
| `nccl_ag` | AllGather | `ncclAllGather` |
| `nccl_ar` | AllReduce | `ncclAllReduce` |
| `nccl_rs` | ReduceScatter | `ncclReduceScatter` |
| `nccl_a2a` | AllToAll | grouped `ncclSend`/`ncclRecv` |
| `nccl_agv` | AllGatherV | grouped broadcast, one root at a time |
| `nccl_a2av` | AllToAllV | grouped variable-count send/receive |
| `nccl_rsv` | ReduceScatterV | grouped reduce, one destination segment at a time |

NCCL does not expose native AllGatherV, AllToAllV, or ReduceScatterV operations;
those targets are semantic emulations.

The NCCL-only targets link CUDA, MPI, NCCL, and MatX. Byte collectives
materialize their deterministic expected output, and reductions compute a
reference output; MatX compares every result with its reference. These targets
do not initialize or link Purlin, NVSHMEM, or mathDx. MPI node-local rank
determines the CUDA device; `CUDA_VISIBLE_DEVICES` can be used to control device
ordering.

## Build and run

Configure and build the suite in Release mode:

```sh
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target \
  testAG testAGV testA2A testA2AV testAR testRS testRSV \
  nccl_ag nccl_agv nccl_a2a nccl_a2av nccl_ar nccl_rs nccl_rsv
```

Each executable accepts the same positional arguments:

```text
<program> [minBytes] [maxBytes] [graphLaunches] [runs] [warmup]
```

Sizes may use `K`, `M`, or `G` suffixes and the endpoints must be powers of two.
For example, a short two-GPU stream-mode run is:

```sh
mpirun -n 2 ./cmake-build-release/nccl_ar 128 1M 0 20 5
```

Purlin targets use NVSHMEM's MPI bootstrap:

```sh
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  mpirun -n 2 ./cmake-build-release/testAR 128 1M 0 20 5
```

To use CUDA Graphs, set `graphLaunches` above zero:

```sh
mpirun -n 2 ./cmake-build-release/nccl_ar 128 1M 4 20 5
```

In graph mode, `runs` collective invocations are captured into one graph. One
graph launch is used for warmup, then `graphLaunches` graph launches are timed.
Reported latency is divided by `runs * graphLaunches`. The `warmup` argument is
used only in stream mode.

## Output and correctness

Rank zero writes CSV. `totalBytes` is the aggregate payload across the world;
there are no separate local-size or maximum-peer-size columns. Algorithm
bandwidth retains the logical payload definition for each operation. Ordinary
Purlin targets print only `lat(us)` and `bw(GB/s)` for the requested collective;
variable-length tests do not run a fixed-count performance comparison. Latency
and bandwidth are limited to four fractional digits.

Every size has an untimed correctness run. Purlin byte collectives use NCCL to
produce their reference buffer. Purlin reduction tests use randomized
floating-point inputs; their reference kernel accumulates sources in Purlin's
guaranteed `0 -> 1 -> ... -> world - 1` order.
MatX performs the final buffer comparison in every case. Each reduction TU
selects its element type through its global `DataType` alias. The `error(%)`
column is the maximum mismatch percentage observed by any rank.
