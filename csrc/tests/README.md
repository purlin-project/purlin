# Collective performance benchmarks

Benchmarks and correctness checks for Purlin collectives. Tests and examples
share argument parsing, timing, data generation, buffers, and CSV reporting in
[`support/purlin/benchmark/`](../support/purlin/benchmark/).

Collective tests use `bench::PurlinRuntime`, which creates a managed context with
`purlin::initialize(rank, world, stream, purlin::NvshmemMemory{})` and releases it
with `purlin::finalize(managed, stream)`. The provider is explicitly included from
[`contrib/symm_mem.cuh`](../contrib/symm_mem.cuh).
The four collective examples use the same runtime helper and show device kernels
with explicit launch configurations. Copy-engine and point-to-point examples use
`purlin::NvshmemMemory` directly for symmetric buffers and peer mappings.

Set `PURLIN_STAGING_TR_SIZE` to change the tests' staging allocation, for example
`32M`. Set `PURLIN_DISABLE_MULTIMEM=1` to exercise unicast paths, or
`PURLIN_DISABLE_MULTIMEM_LR=1` to disable multicast for latency staging only.

## Targets

| Operation | Fixed counts | Variable counts | Extra skew cases |
| --- | --- | --- | --- |
| AllGather | `testAG` | `testAGV` | `testAGVSKEW` |
| AllReduce | `testAR` | — | — |
| ReduceScatter | `testRS` | `testRSV` | `testRSVSKEW` |
| AllToAll | `testA2A` | `testA2AV` | `testA2AVSKEW` |

## Build and run

Run from `csrc`:

```sh
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build cmake-build-release --target \
  testAG testAGV testA2A testA2AV testAR testRS testRSV \
  testAGVSKEW testA2AVSKEW testRSVSKEW
```

Benchmark arguments:

```text
<program> [minBytes] [maxBytes] [graphLaunches] [runs] [warmup] [seed]
```

- Sizes accept `K`, `M`, or `G` suffixes; endpoints must be powers of two.
- `graphLaunches=0` times `runs` stream invocations after `warmup` calls.
  A positive value times that many CUDA Graph launches, with `runs` calls per
  graph and one warmup launch; `warmup` is ignored. Latency is per invocation.
- `seed=0` (the default) chooses a random seed. Reuse the seed printed on stderr
  to reproduce the inputs.

A two-GPU stream run using NVSHMEM's MPI bootstrap:

```sh
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  mpirun -n 2 ./cmake-build-release/testAR 128 1M 0 20 5
```

Reduction tests (`testAR`, `testRS`, `testRSV`, `testRSVSKEW`, and
`testVarlenEmpty`) also accept `--reduction-mode deterministic|non-deterministic`
anywhere in the command. The default, `non-deterministic`, allows multimem when
available; `deterministic` uses rank-ordered unicast reduction.

## Output and correctness

Rank zero writes CSV with payload size, per-invocation latency, and bandwidth.
Every size gets an untimed correctness check against locally generated reference
data, compared with MatX. Reduction benchmarks use a rank-ordered reference.
Non-deterministic mode on `ARCH >= 900` uses small integers whose sums are exact,
so different reduction orders still compare exactly. Other modes and older
architectures use seeded random inputs.

## Variable-count split policy

`testAGV`, `testRSV`, and `testA2AV` sweep **total bytes**, doubling each step:

- AGV: input partitions across ranks sum to the total.
- RSV: output partitions sum to the total; each rank supplies the full input.
- A2AV: each rank sends and receives the total.

CSV `totalBytes` and bandwidth use that same total. Splits are reproducible,
128-byte aligned, and sum exactly to the total. They use a mild Zipf skew by
default; set `BENCH_ZIPF_EXPONENT` to `0` (equal), `0.125` (default), `0.25`, `0.5`,
or `1` for increasing skew. Totals must be positive multiples of 128 bytes, with
at most 32 ranks. Small totals can produce empty partitions.

See [`variable_counts.cuh`](../support/purlin/benchmark/variable_counts.cuh)
for rounding and rank placement. Separate `*SKEW` targets use their own policies.

## Regression checks

- `testSetup` checks managed ownership, zeroed workspace regions, custom staging
  sizes, provider lifetime, validation, multicast controls, and caller-owned
  finalization. It uses a single-GPU test provider and builds without NVSHMEM.
- `testSetupNvshmem` checks repeated managed initialization and finalization,
  zeroed peer mappings, and peer writes using the supplied NVSHMEM provider.
- `testStaticFor` checks compile-time indices, empty and nested loops, ordered
  execution, and callbacks that cannot be copied. It runs host and single-GPU
  checks without an MPI launch.
- `testVariableCounts` checks split totals, alignment, send/receive consistency,
  and invalid inputs. It runs without GPUs or an MPI launch.
- `testVarlenEmpty` checks empty partitions, delayed peers, and context reuse
  across sizes and collectives in stream and CUDA Graph modes. Changing inputs
  on each invocation catches stale data as well as hangs.

After configuring as above:

```sh
cmake --build cmake-build-release --target testStaticFor testVariableCounts testVarlenEmpty
./cmake-build-release/testStaticFor
./cmake-build-release/testVariableCounts
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  timeout 180s mpirun -n 2 ./cmake-build-release/testVarlenEmpty
```

Run the setup checks with:

```sh
cmake --build cmake-build-release --target testSetup testSetupNvshmem
./cmake-build-release/testSetup
PURLIN_DISABLE_MULTIMEM=1 ./cmake-build-release/testSetup
PURLIN_DISABLE_MULTIMEM_LR=1 ./cmake-build-release/testSetup
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  timeout 180s mpirun -n 2 ./cmake-build-release/testSetupNvshmem
```

## Examples

The device-kernel examples (`ag`, `a2a`, `ar`, `rs`, `push`, and `pull`) use:

```text
<program> [minBytes] [maxBytes] [maxBlocks] [graphLaunches] [runs] [warmup] [seed]
```

`maxBlocks` is the existing tuning limit: superblock size for `ag`, `a2a`, `push`,
and `pull`, or reduction blocks for `ar` and `rs`. Kernel configurations and CSV
tuning columns are defined in each example. `ar` and `rs` also accept
`--reduction-mode deterministic|non-deterministic` anywhere in the command and
use the tests' input policy: predictable values only when `ARCH >= 900` and the
mode is non-deterministic. Both examples currently configure unicast kernels,
which provide rank-ordered reductions in either mode.

`ce_p2p` uses the tests' argument order, with defaults of 16 runs and 16 warmups.
`ce_ag` keeps its original argument order and defaults:

```text
ce_ag [minBytes] [maxBytes] [warmup=128] [runs=256] [graphLaunches=2]
```

All examples share stream/graph timing, size sweeps, seeded data generation, and
correctness helpers with the tests. One-way transfers report the issuing rank's
latency and the receiving rank's correctness result. `error(%)` is a percentage
in every example, and graph warmup reports one captured batch (`runs` calls).

Build the examples and exercise all-to-all across its latency, throughput, and
chunked paths in both stream and graph modes. Its grid must stay within
`MAX_NUM_CTAS`:

```sh
cmake --build cmake-build-release --target ag a2a ar rs ce_ag ce_p2p push pull
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  timeout 180s mpirun -n 2 ./cmake-build-release/a2a 128 8M 32 0 2 1
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  timeout 180s mpirun -n 2 ./cmake-build-release/a2a 128 8M 32 2 2 1
```
