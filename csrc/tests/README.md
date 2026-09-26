# Collective performance benchmarks

Benchmarks and correctness checks for Purlin collectives. Tests and examples
share argument parsing, timing, data generation, buffers, and CSV reporting in
[`support/purlin/benchmark/`](../support/purlin/benchmark/).

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
data, compared with MatX. Deterministic reductions use seeded random inputs and
a rank-ordered reference; non-deterministic reductions use small integers whose
sums are exact, so different reduction orders still compare exactly.

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

- `testVariableCounts` checks split totals, alignment, send/receive consistency,
  and invalid inputs. It runs without GPUs or an MPI launch.
- `testVarlenEmpty` checks empty partitions, delayed peers, and context reuse
  across sizes and collectives in stream and CUDA Graph modes. Changing inputs
  on each invocation catches stale data as well as hangs.

After configuring as above:

```sh
cmake --build cmake-build-release --target testVariableCounts testVarlenEmpty
./cmake-build-release/testVariableCounts
NVSHMEM_BOOTSTRAP=MPI NVSHMEM_REMOTE_TRANSPORT=none \
  timeout 180s mpirun -n 2 ./cmake-build-release/testVarlenEmpty
```
